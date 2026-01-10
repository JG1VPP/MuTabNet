import torch
import torch.nn as nn
import torch.nn.functional as F
from more_itertools import transpose

from mutab.model.factory import MODELS, build


@MODELS.register_module()
class TableHandler(nn.Module):
    def __init__(
        self,
        html_dict: dict,
        cell_dict: dict,
        SOC: str,
        revisor: dict,
    ):
        super().__init__()

        assert isinstance(html_dict, dict)
        assert isinstance(cell_dict, dict)

        assert isinstance(SOC, str)

        self.SOC = SOC

        self.html = build(html_dict)
        self.cell = build(cell_dict)

        self.revisor = build(revisor)

    @property
    def SOC_HTML(self):
        token = self.SOC

        token = self.html.get_number(token)
        token = self.html.get_tensor(token)

        return token

    def forward(self, img_metas, train: bool):
        if train:
            return self._train(img_metas)
        else:
            return self._valid(img_metas)

    def reverse(self, img_metas, **items):
        outputs = []

        for item in transpose(items.values()):
            outputs.append(dict(zip(items.keys(), item)))

        return tuple(map(self.itemize, outputs, img_metas))

    def itemize(self, outputs, targets):
        # decode
        outputs.update(bbox=self.decode_bbox(**outputs))
        outputs.update(html=self.decode_html(**outputs))
        outputs.update(cell=self.decode_cell(**outputs))

        # revise
        outputs.update(real=self.revisor(**targets))
        outputs.update(pred=self.revisor(**outputs))

        return dict(outputs)

    def _train(self, batch):
        html = self.tensor("html", batch, self.encode_html)
        cell = self.tensor("cell", batch, self.encode_cell)
        bbox = self.tensor("bbox", batch, self.encode_bbox)

        # align
        bbox = F.pad(bbox, pad=(1, 1)).mT

        # batch
        item = dict(targets=batch)

        # tasks
        item.update(html=html)
        item.update(back=html.fliplr())
        item.update(cell=cell)
        item.update(bbox=bbox)

        return item

    def _valid(self, batch):
        # batch
        return dict(targets=batch)

    def tensor(self, key, batch, op):
        data = op(m.get(key) for m in batch)
        size = max(v.size(-1) for v in data)

        def op(x):
            pad = (0, size - x.size(-1))
            return F.pad(x, pad=pad)

        data = torch.stack(list(map(op, data)))
        return data.to(device=self.html.device)

    def encode_html(self, batch):
        batch = map(self.html.enclose, batch)
        batch = map(self.html.forward, batch)

        return tuple(batch)

    def encode_cell(self, batch):
        batch = map(self.cell.flatten, batch)
        batch = map(self.cell.enclose, batch)
        batch = map(self.cell.forward, batch)

        return tuple(batch)

    def encode_bbox(self, batch):
        batch = map(torch.from_numpy, batch)
        return tuple(box.T for box in batch)

    def decode_html(self, html, **kwargs):
        return self.html.reverse(html)

    def decode_cell(self, cell, **kwargs):
        cell = self.cell.reverse(cell)
        return self.cell.resolve(cell)

    def decode_bbox(self, bbox, html, **kwargs):
        return bbox[html.eq(self.SOC_HTML)].cpu().numpy()
