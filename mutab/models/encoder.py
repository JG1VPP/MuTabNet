from positional_encodings import torch_encodings as pos

from mutab.models.factory import ENCODERS


@ENCODERS.register_module()
class PositionalEncoding2D(pos.PositionalEncodingPermute2D):
    def forward(self, img):
        return super().forward(img).add(img).flatten(2).mT
