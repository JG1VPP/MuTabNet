from pathlib import Path

import numpy as np
from mmocr.datasets.builder import LOADERS, PARSERS, build_parser

from mutab.utils import get_logger


@PARSERS.register_module()
class TableStrParser:
    def __init__(self, cell_tokens, empty_bbox):
        assert isinstance(cell_tokens, list)
        assert isinstance(empty_bbox, tuple)
        self.cell_tokens = cell_tokens
        self.empty_bbox = empty_bbox

    def align(self, html, bbox, **info):
        queue = iter(bbox)
        boxes = np.zeros((len(html), 4))
        for idx, cell in enumerate(html):
            if cell in self.cell_tokens:
                boxes[idx] = next(queue)
        return dict(html=html, bbox=boxes, **info)

    def parse(self, ann: str, **info):
        list_b = []
        list_c = []

        with open(ann) as f:
            path = f.readline().strip()
            html = f.readline().strip().split(",")
            body = list(f.readlines())

        for value in body:
            bbox, cell = value.strip().split("<;>")
            bbox = tuple(map(int, bbox.split(",")))
            cell = cell.split("\t")
            list_b.append(bbox)
            if bbox != self.empty_bbox:
                list_c.append(cell)

        info.update(filename=path, html=html)
        info.update(cell=list_c, bbox=list_b)

        return info

    def __call__(self, info):
        return self.align(**self.parse(**info))


@LOADERS.register_module()
class TableHardDiskLoader:
    def __init__(self, parser: dict, ann_file: str):
        self.infos = self.load(parser, ann_file)

    def __len__(self):
        return len(self.infos)

    def __getitem__(self, index):
        return self.infos[index]

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx < len(self):
            data = self[self.idx]
            self.idx += 1
            return data
        raise StopIteration

    def load(self, parser: dict, ann_file: str):
        parser = build_parser(parser)
        tables = []
        logger = get_logger()
        logger.info(f"Loading {ann_file} ...")
        for f in Path(ann_file).rglob("*.txt"):
            tables.append(parser(dict(ann=f)))
        logger.info(f"{len(tables)} tables")
        return tables
