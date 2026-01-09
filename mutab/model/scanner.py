from collections import ChainMap
from functools import partial
from time import perf_counter_ns

import torch.distributed as dist
import torch.nn as nn
from mmcv.image import imread
from mmcv.runner import BaseModule, auto_fp16
from mmocr.apis import model_inference

from mutab.model.factory import MODELS, build


@MODELS.register_module()
class TableScanner(BaseModule):
    def __init__(
        self,
        backbone,
        encoder,
        decoder,
        handler,
        html_loss,
        cell_loss,
        **kwargs,
    ):
        super().__init__()

        # label handler
        assert handler is not None
        self.handler = build(handler)

        # backbone
        assert backbone is not None
        self.backbone = build(backbone)

        # encoder module
        assert encoder is not None
        self.encoder = build(encoder)

        # decoder module
        assert decoder is not None
        decoder.update(num_emb_html=self.handler.num_class_html)
        decoder.update(num_emb_cell=self.handler.num_class_cell)

        # special tokens (html)
        decoder.update(SOC_HTML=self.handler.SOC_HTML)
        decoder.update(SOS_HTML=self.handler.SOS_HTML)
        decoder.update(EOS_HTML=self.handler.EOS_HTML)

        # special tokens (cell)
        decoder.update(SOS_CELL=self.handler.SOS_CELL)
        decoder.update(EOS_CELL=self.handler.EOS_CELL)
        decoder.update(SEP_CELL=self.handler.SEP_CELL)

        self.decoder = build(decoder)

        # loss
        assert isinstance(html_loss, list) and len(html_loss)
        assert isinstance(cell_loss, list) and len(cell_loss)

        pad_html = partial(build, ignore=self.handler.PAD_HTML)
        pad_cell = partial(build, ignore=self.handler.PAD_CELL)

        self.loss = nn.ModuleList()
        self.loss.extend(tuple(map(pad_html, html_loss)))
        self.loss.extend(tuple(map(pad_cell, cell_loss)))

        self.init_weights()

    @property
    def is_init(self):
        return True

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @auto_fp16(apply_to=["img"])
    def forward(self, img, img_metas, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(img, img_metas)
        elif isinstance(img_metas[0], list):
            return self.forward_test(img, img_metas[0])
        else:
            return self.forward_test(img, img_metas)

    def train_step(self, data, optimizer):
        loss = self.parse_losses(self(**data))
        loss.update(num_samples=len(data["img_metas"]))
        return loss

    def val_step(self, data, optimizer):
        loss = self.parse_losses(self(**data))
        loss.update(num_samples=len(data["img_metas"]))
        return loss

    def parse_losses(self, losses):
        logs = dict({k: v.mean() for k, v in losses.items()})
        loss = sum(v for k, v in logs.items() if "loss" in k)
        logs.update(loss=loss)
        for key, value in logs.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                value = value.data.clone()
                world = int(dist.get_world_size())
                dist.all_reduce(value.div_(world))
            logs[key] = value.item()
        return dict(loss=loss, log_vars=logs)

    def forward_train(self, image, img_metas):
        targets = self.handler.forward(img_metas, device=image.device)
        outputs = self.decoder(self.encoder(self.backbone(image)), **targets)
        return ChainMap(*tuple(loss(outputs, targets) for loss in self.loss))

    def forward_test(self, image, img_metas):
        return self.simple_test(image, img_metas)

    def simple_test(self, image, img_metas):
        time = perf_counter_ns()
        item = self.decoder.predict(self.encoder(self.backbone(image)), time)
        return tuple(self.handler.reverse(**item, img_metas=list(img_metas)))

    def predict(self, path: str):
        return dict(path=path, **model_inference(self, imread(path)))
