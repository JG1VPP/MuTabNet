import glob
import os

import numpy as np
from mmocr.datasets.builder import LOADERS, PARSERS, build_parser

from mutab.utils import get_logger


@PARSERS.register_module()
class TableStrParser:
    def __init__(self, cell_tokens):
        assert isinstance(cell_tokens, list)
        assert len(cell_tokens)
        self.cell_tokens = cell_tokens

    def align(self, html, bbox, **info):
        queue = iter(bbox)
        boxes = np.zeros((len(html), 4))
        for idx, cell in enumerate(html):
            if cell in self.cell_tokens:
                boxes[idx] = next(queue)
        return dict(html=html, bbox=boxes, **info)

    def __call__(self, info):
        return self.align(**info)


@LOADERS.register_module()
class TableHardDiskLoader:
    def __init__(self, parser: dict, ann_file: str, max_len_html: int):
        self.parser = build_parser(parser)
        self.infos = self.load(ann_file, max_len_html)

    def __len__(self):
        return len(self.infos)

    def __getitem__(self, index):
        return self.parser(self.infos[index])

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx < len(self):
            data = self[self.idx]
            self.idx += 1
            return data
        raise StopIteration

    def load(self, ann_file: str, max_len_html: int):
        data = []
        logger = get_logger()
        logger.info(f"Loading {ann_file} ...")
        for f in glob.glob(os.path.join(ann_file, "*.txt")):
            with open(f) as f:
                data.append(self.parse(f))
        logger.info(f"{len(data)} tables were loaded from {ann_file}")
        return list(v for v in data if len(v["html"]) <= max_len_html)

    def parse(self, f):
        path = f.readline().strip()
        html = f.readline().strip().split(",")
        bbox_list = []
        cell_list = []
        for value in f.readlines():
            bbox, cell = value.strip().split("<;>")
            bbox = tuple(map(int, bbox.split(",")))
            bbox_list.append(bbox)
            if bbox != (0, 0, 0, 0):
                cell_list.append(cell.split("\t"))
        return dict(filename=path, html=html, cell=cell_list, bbox=bbox_list)
