from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np
from mmcv.transforms import BaseTransform
from mmengine import TRANSFORMS
from mmengine.structures import BaseDataElement
from more_itertools import chunked, collapse, transpose

from mutab.table import html_to_otsl, otsl_to_html

EASY = "simple"
HARD = "complex"


class TableTransform(ABC, BaseTransform):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def transform(self, results):
        return results | self.progress(**results)

    @abstractmethod
    def progress(self, **kwargs) -> dict:
        raise NotImplementedError


@TRANSFORMS.register_module()
class Annotate(BaseTransform):
    def __init__(self, *, keys, meta):
        super().__init__()

        assert isinstance(keys, Sequence)
        assert isinstance(meta, Sequence)

        self.keys = keys
        self.meta = meta

    def transform(self, results):
        # collect inputs
        data = {k: results[k] for k in self.keys}
        meta = {k: results[k] for k in self.meta}

        # create element
        meta = BaseDataElement(metainfo=meta)

        return dict(data, targets=meta)


@TRANSFORMS.register_module()
class FillBbox(TableTransform):
    def __init__(self, cell: list[str], **kwargs):
        super().__init__()

        self.cell = cell

    def progress(self, html, cell, **kwargs):
        html_bbox = np.zeros((len(html), 4))

        list_cell = list(v["text"] for v in cell)
        iter_bbox = iter(v["bbox"] for v in cell)

        for idx, cell in enumerate(html):
            if cell in self.cell:
                html_bbox[idx] = next(iter_bbox)

        return dict(cell=list_cell, gt_bboxes=html_bbox)


@TRANSFORMS.register_module()
class FormBbox(TableTransform):
    def progress(self, img_shape, gt_bboxes, **kwargs):
        img_h, img_w = img_shape

        # lurd format
        l = gt_bboxes[..., 0] / img_w
        u = gt_bboxes[..., 1] / img_h
        r = gt_bboxes[..., 2] / img_w
        d = gt_bboxes[..., 3] / img_h

        # xywh format
        x = (l + r) / 2
        y = (u + d) / 2
        w = abs(r - l)
        h = abs(d - u)

        # must be ndarray
        bbox = np.stack([x, y, w, h], axis=-1)
        return dict(bbox=bbox, gt_bboxes=bbox)


@TRANSFORMS.register_module()
class Hardness(TableTransform):
    def progress(self, html, **kwargs):
        return dict(type=(EASY, HARD)[">" in html])


@TRANSFORMS.register_module()
class ToOTSL(TableTransform):
    def progress(self, html, bbox, **kwargs):
        otsl, bbox, _, _ = html_to_otsl(html, bbox)
        return dict(html=otsl, bbox=np.stack(bbox))


@TRANSFORMS.register_module()
class ToVTML(TableTransform):
    def progress(self, html, bbox, **kwargs):
        otsl, _, rows, cols = html_to_otsl(html, bbox)

        otsl = chunked(map(self.switch, otsl), n=cols)
        otsl = list(map(self.bottom, transpose(otsl)))
        vtml = otsl_to_html(list(collapse(otsl[:-1])))

        return dict(vtml=vtml)

    def switch(self, cell):
        if cell == "L":
            return "U"

        if cell == "U":
            return "L"

        return cell

    def bottom(self, cells):
        return (*cells, "R")
