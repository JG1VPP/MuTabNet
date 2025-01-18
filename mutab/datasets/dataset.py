import numpy as np
from mmdet.datasets.builder import DATASETS
from mmocr.datasets import BaseDataset

from mutab.metrics import TEDS
from mutab.utils import get_logger


@DATASETS.register_module()
class TableDataset(BaseDataset):
    def evaluate(self, results, **kwargs):
        scores = []
        logger = get_logger()
        for idx, table in enumerate(self.data_infos):
            score = TEDS(struct_only=False).evaluate(**results[idx])
            logger.info("TEDS: %.3f (%s)", score, table["filename"])
            scores.append(score)

        return dict(TEDS=np.mean(scores))
