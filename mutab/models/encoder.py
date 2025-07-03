import torch.nn as nn
from positional_encodings import torch_encodings as pos

from mutab.models.attention import Blocks
from mutab.models.factory import ENCODERS


@ENCODERS.register_module()
class TableEncoder(nn.Module):
    def __init__(self, d_model: int, **kwargs):
        super().__init__()

        # blocks
        self.pos = pos.PositionalEncoding2D(d_model)
        self.enc = Blocks(d_model=d_model, **kwargs)

    def forward(self, img):
        return self.process(img.permute(0, 2, 3, 1))

    def process(self, img):
        assert img.ndim == 4

        # forward
        img = self.pos(img).add(img)
        hid = self.enc(x=img, y=img)

        return hid
