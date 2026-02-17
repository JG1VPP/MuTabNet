from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from more_itertools import transpose

from mutab.utils import MODELS, build


@MODELS.register_module()
class TableHandler(nn.Module):
    def __init__(
        self,
        html_dict: dict,
        cell_dict: dict,
        SOC: list[str],
        revisor: dict,
        outputs: list[str],
        targets: list[str],
    ):
        super().__init__()

        assert isinstance(html_dict, dict)
        assert isinstance(cell_dict, dict)

        self.html = build(html_dict)
        self.cell = build(cell_dict)

        self.revisor = build(revisor)

        assert isinstance(outputs, Sequence)
        assert isinstance(targets, Sequence)

        self.outputs = outputs
        self.targets = targets

        assert isinstance(SOC, Sequence)

        self.register_buffer("SOC_HTML", self.special(SOC))

    def special(self, token):
        token = map(self.html.get_number, token)
        token = map(self.html.get_tensor, token)

        return torch.stack(tuple(token))

    def forward(self, img, targets, train: bool):
        if train:
            return self._train(img, targets)
        else:
            return self._valid(img, targets)

    def reverse(self, targets, **items):
        outputs = []

        for item in transpose(items.values()):
            outputs.append(dict(zip(items.keys(), item)))

        return tuple(map(self.itemize, outputs, targets))

    def itemize(self, outputs, targets):
        targets = targets.to_dict()

        # decode
        outputs.update(bbox=self.decode_bbox(**outputs))
        outputs.update(html=self.decode_html(**outputs))
        outputs.update(cell=self.decode_cell(**outputs))

        # revise
        targets.update(full=self.revisor(**targets))
        outputs.update(full=self.revisor(**outputs))

        # labels
        outputs = {k: outputs.get(k) for k in self.outputs}
        targets = {k: targets.get(k) for k in self.targets}

        return dict(outputs=outputs, targets=targets)

    def _train(self, img, batch):
        html = self.tensor("html", batch, self.encode_html)
        vtml = self.tensor("vtml", batch, self.encode_html)
        cell = self.tensor("cell", batch, self.encode_cell)
        bbox = self.tensor("bbox", batch, self.encode_bbox)

        assert html.ne(self.html.UKN).all().item()

        # align
        bbox = F.pad(bbox, pad=(1, 1)).mT

        # batch
        item = dict(targets=batch, img=torch.stack(img))

        # tasks
        item.update(html=html)
        item.update(back=html.fliplr())
        item.update(vtml=vtml)
        item.update(flip=vtml.fliplr())
        item.update(cell=cell)
        item.update(bbox=bbox)
        item.update(zone=bbox)

        return item

    def _valid(self, img, batch):
        # batch
        return dict(targets=batch, img=torch.stack(img))

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
        return bbox[torch.isin(html, self.SOC_HTML)].cpu().numpy()
