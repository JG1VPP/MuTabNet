from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np
from mmcv.transforms import BaseTransform
from mmengine import TRANSFORMS
from mmengine.structures import BaseDataElement

from mutab.table import html_to_otsl

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
class FlipBbox(TableTransform):
    def progress(self, img_shape, bbox, **kwargs):
        h, _ = img_shape

        mask = np.count_nonzero(bbox, axis=-1, keepdims=True)
        flip = np.where(mask, h - 1 - bbox, bbox).clip(min=0)

        np.copyto(bbox[..., 1], flip[..., 1])
        np.copyto(bbox[..., 3], flip[..., 3])

        return dict(bbox=bbox)


@TRANSFORMS.register_module()
class FormBbox(TableTransform):
    def progress(self, img_shape, gt_bboxes, **kwargs):
        bb = np.zeros_like(gt_bboxes)

        # xyxy to xy
        bb[..., 0] = gt_bboxes[..., 0::2].mean(axis=-1)
        bb[..., 1] = gt_bboxes[..., 1::2].mean(axis=-1)

        # xyxy to wh
        bb[..., 2] = np.ptp(gt_bboxes[..., 0::2], axis=-1)
        bb[..., 3] = np.ptp(gt_bboxes[..., 1::2], axis=-1)

        # normalize
        bb[..., 0::2] /= img_shape[1]
        bb[..., 1::2] /= img_shape[0]

        assert np.all(bb >= 0)
        assert np.all(bb <= 1)

        assert np.any(bb > 0)
        assert np.any(bb < 1)

        return dict(bbox=bb)


@TRANSFORMS.register_module()
class Hardness(TableTransform):
    def progress(self, html, **kwargs):
        return dict(type=(EASY, HARD)[">" in html])


@TRANSFORMS.register_module()
class ToOTSL(TableTransform):
    def progress(self, html, bbox, **kwargs):
        otsl, bbox, _, n = html_to_otsl(html, bbox)
        return dict(html=otsl, bbox=np.stack(bbox))
