from pathlib import Path
from typing import List

from mmengine.dataset import BaseDataset
from mmengine.fileio import load
from mmengine.registry import DATASETS


@DATASETS.register_module()
class TableDataset(BaseDataset):
    @property
    def split(self):
        return self.filter_cfg.get("split")

    def load_data_list(self) -> List[dict]:
        # load preprocessed pickle
        path = Path(self.ann_file).expanduser()
        data = load(path, file_format="pickle")

        return list(data[self.split])
