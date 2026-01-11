from collections import ChainMap
from functools import partial

import torch
import torch.nn as nn
from mmengine.model import BaseModel

from mutab.utils import MODELS, build


@MODELS.register_module()
class TableScanner(BaseModel):
    def __init__(
        self,
        encoder,
        decoder,
        handler,
        html_loss,
        cell_loss,
        **kwargs,
    ):
        super().__init__()

        # handler module
        self.handler = build(handler)

        # encoder module
        self.encoder = build(encoder)

        # decoder module
        decoder.update(num_emb_html=self.handler.html.num_class)
        decoder.update(num_emb_cell=self.handler.cell.num_class)

        # special tokens (html)
        decoder.update(SOC_HTML=self.handler.SOC_HTML)
        decoder.update(SOS_HTML=self.handler.html.SOS)
        decoder.update(EOS_HTML=self.handler.html.EOS)

        # special tokens (cell)
        decoder.update(SOS_CELL=self.handler.cell.SOS)
        decoder.update(EOS_CELL=self.handler.cell.EOS)
        decoder.update(SEP_CELL=self.handler.cell.SEP)

        self.decoder = build(decoder)

        # loss
        assert isinstance(html_loss, list) and len(html_loss)
        assert isinstance(cell_loss, list) and len(cell_loss)

        pad_html = partial(build, ignore=self.handler.html.PAD.item())
        pad_cell = partial(build, ignore=self.handler.cell.PAD.item())

        self.losses = nn.ModuleList()
        self.losses.extend(tuple(map(pad_html, html_loss)))
        self.losses.extend(tuple(map(pad_cell, cell_loss)))

    def init_weights(self):
        pass

    def forward(self, mode: str, **kwargs):
        if mode == "loss":
            return self._train(**kwargs)

        with torch.no_grad():
            return self._valid(**kwargs)

    def _train(self, **targets):
        targets = self.handler(**targets, train=True)
        outputs = self.encoder(**targets, train=True)
        outputs = self.decoder(**outputs, train=True)

        return self.loss(outputs, targets)

    def _valid(self, **targets):
        outputs = self.handler(**targets, train=False)
        outputs = self.encoder(**outputs, train=False)
        outputs = self.decoder(**outputs, train=False)

        return self.handler.reverse(**outputs, **targets)

    def loss(self, outputs, targets):
        return ChainMap(*[f(outputs, targets) for f in self.losses])
