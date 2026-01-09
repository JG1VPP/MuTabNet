from collections import defaultdict
from functools import cached_property, partial
from itertools import product
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from more_itertools import flatten, split_at

from mutab.model.factory import MODELS, build


@MODELS.register_module()
class TableHandler(nn.Module):
    def __init__(
        self,
        html_dict_file: str,
        cell_dict_file: str,
        SOC: List[str],
        revisor: Dict[str, str],
    ):
        super().__init__()

        assert isinstance(html_dict_file, str)
        assert isinstance(cell_dict_file, str)

        assert isinstance(SOC, list)

        self.SOC = SOC

        self.char2idx_html, self.idx2char_html = self.load(html_dict_file)
        self.char2idx_cell, self.idx2char_cell = self.load(cell_dict_file)

        self.SOS_HTML = self.add(self.char2idx_html, self.idx2char_html, "<SOS>")
        self.EOS_HTML = self.add(self.char2idx_html, self.idx2char_html, "<EOS>")
        self.PAD_HTML = self.add(self.char2idx_html, self.idx2char_html, "<PAD>")
        self.UKN_HTML = self.add(self.char2idx_html, self.idx2char_html, "<UKN>")

        self.SOS_CELL = self.add(self.char2idx_cell, self.idx2char_cell, "<SOS>")
        self.EOS_CELL = self.add(self.char2idx_cell, self.idx2char_cell, "<EOS>")
        self.PAD_CELL = self.add(self.char2idx_cell, self.idx2char_cell, "<PAD>")
        self.SEP_CELL = self.add(self.char2idx_cell, self.idx2char_cell, "<SEP>")
        self.UKN_CELL = self.add(self.char2idx_cell, self.idx2char_cell, "<UKN>")

        assert len(self.char2idx_html) == len(self.idx2char_html)
        assert len(self.char2idx_cell) == len(self.idx2char_cell)

        self.char2idx_html = defaultdict(lambda: self.UKN_HTML, self.char2idx_html)
        self.char2idx_cell = defaultdict(lambda: self.UKN_CELL, self.char2idx_cell)

        self.revisor = build(revisor)

    def load(self, dict_file: str, enc="utf-8"):
        with open(dict_file, encoding=enc) as f:
            idx2char = list(filter(None, f.read().splitlines()))
            char2idx = dict(zip(idx2char, range(len(idx2char))))
        return char2idx, idx2char

    def add(self, char2idx, idx2char, token: str):
        idx = len(idx2char)
        idx2char.append(token)
        char2idx[token] = idx
        return idx

    @property
    def num_class_html(self):
        return len(self.idx2char_html)

    @property
    def num_class_cell(self):
        return len(self.idx2char_cell)

    @cached_property
    def SOC_HTML(self):
        return list(self.char2idx_html[v] for v in self.SOC)

    def str2idx(self, strings, char2idx):
        return list([char2idx[v] for v in sample] for sample in strings)

    def idx2str(self, indices, idx2char):
        return list([idx2char[i] for i in sample] for sample in indices)

    def pad_tensor(self, batch, value):
        pad = lambda seq, size: F.pad(seq, (0, size - len(seq)), value=value)
        return torch.stack([pad(seq, max(map(len, batch))) for seq in batch])

    def encode_html(self, batch):
        samples = []
        for idx in self.str2idx(batch, self.char2idx_html):
            idx = (self.SOS_HTML, *idx, self.EOS_HTML)
            samples.append(torch.tensor(idx))
        return self.pad_tensor(samples, self.PAD_HTML)

    def encode_cell(self, batch):
        samples = []
        sos = self.SOS_CELL
        eos = self.EOS_CELL
        sep = self.SEP_CELL
        for sample in batch:
            item = self.str2idx(sample, self.char2idx_cell)
            item = flatten(flatten(product(item, [[sep]])))
            samples.append(torch.tensor([sos, *item, eos]))
        return self.pad_tensor(samples, self.PAD_CELL)

    def decode_html(self, batch):
        strip = lambda it: next(split_at(it, lambda n: n == self.EOS_HTML))
        return self.idx2str(map(strip, batch.tolist()), self.idx2char_html)

    def decode_cell(self, batch):
        strings = []
        for idx in batch.tolist():
            idx = next(split_at(idx, lambda n: n == self.EOS_CELL))
            idx = list(split_at(idx, lambda n: n == self.SEP_CELL))
            strings.append(self.idx2str(idx, self.idx2char_cell))
        return strings

    def encode_bbox(self, batch):
        pad = lambda bb, k: F.pad(torch.from_numpy(bb), (0, 0, 1, k - len(bb)))
        return torch.stack([pad(bb, 1 + max(map(len, batch))) for bb in batch])

    def decode_bbox(self, batch, mask, img_metas):
        results = []
        for bbox, mask, meta in zip(batch, mask, img_metas):
            bbox = bbox.cpu().numpy()
            mask = mask.cpu().numpy()
            scale = meta["img_scale"]
            shape = meta["pad_shape"]
            bbox[:, 0::2] *= shape[1]
            bbox[:, 1::2] *= shape[0]
            bbox[:, 0::2] /= scale[1]
            bbox[:, 1::2] /= scale[0]
            results.append(bbox[mask])
        return results

    def item(self, html, cell, bbox, img_meta, time):
        results = dict(meta=img_meta)
        results.update(html=html, cell=cell, real=self.revisor(**img_meta))
        results.update(bbox=bbox, time=time, pred=self.revisor(html, cell))
        return results

    def forward(self, img_metas, device):
        html = self.encode_html([m["html"] for m in img_metas]).to(device)
        cell = self.encode_cell([m["cell"] for m in img_metas]).to(device)
        bbox = self.encode_bbox([m["bbox"] for m in img_metas]).to(device)
        return dict(html=html, back=html.fliplr(), cell=cell, bbox=bbox)

    def reverse(self, html, cell, bbox, time, img_metas, **kwargs):
        mask = torch.isin(html, torch.tensor(self.SOC_HTML).to(html))
        bbox = self.decode_bbox(bbox, mask=mask, img_metas=img_metas)
        html = self.decode_html(html)
        cell = self.decode_cell(cell)
        return tuple(map(partial(self.item, time=time), html, cell, bbox, img_metas))
