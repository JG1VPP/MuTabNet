import torch.nn as nn
from positional_encodings import torch_encodings as pos

from mutab.block import Blocks
from mutab.utils import MODELS, build


@MODELS.register_module()
class TableEncoder(nn.Module):
    def __init__(self, backbone: dict, d_model: int, **kwargs):
        super().__init__()

        # backbone
        self.net = build(backbone)

        # blocks
        self.pos = pos.PositionalEncoding2D(d_model)
        self.enc = Blocks(d_model=d_model, **kwargs)

    def forward(self, img, train: bool, **kwargs):
        return dict(kwargs, img=self.process(img))

    def process(self, img):
        assert img.ndim == 4

        # forward
        img = self.net(img).permute(0, 2, 3, 1)
        img = self.pos(img).add(img)
        hid = self.enc(x=img, y=img)

        return hid
