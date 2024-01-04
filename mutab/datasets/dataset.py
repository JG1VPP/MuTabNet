import numpy as np
from mmdet.datasets.builder import DATASETS
from mmocr.datasets import BaseDataset

from mutab.metrics import TEDS
from mutab.utils import get_logger


@DATASETS.register_module()
class TableDataset(BaseDataset):
    def evaluate(self, results, **kwargs):
        metric = TEDS(struct_only=False)
        scores = []
        logger = get_logger()
        for idx, info in enumerate(self.data_infos):
            score = metric.evaluate(**results[idx])
            logger.info("%s score: %s", info["filename"], score)
            scores.append(score)

        return dict(TEDS=np.mean(scores))
