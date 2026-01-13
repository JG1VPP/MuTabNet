import torch.nn as nn

from mutab.block import Linear
from mutab.utils import MODELS


@MODELS.register_module()
class TableCellLocator(Linear):
    def __init__(self, d_model: int, **kwargs):
        super().__init__(d_model, 4, act=nn.Sigmoid)

    def forward(self, img, hid1, hid2):
        box1 = super().forward(hid1)
        box2 = super().forward(hid2)

        return box1, box2
