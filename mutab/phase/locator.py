import torch.nn as nn

from mutab.block import Linear
from mutab.utils import MODELS


@MODELS.register_module()
class TableCellLocator(nn.Module):
    def __init__(self, d_model: int, pass_html: bool, **kwargs):
        super().__init__()

        # embeddings
        self.pos = Linear(d_model, 4, act=nn.Sigmoid)
        self.emb = Linear(4, d_model, act=nn.Sigmoid)

        self.pass_html = int(pass_html)

    def forward(self, img, html, grid):
        plus = self.emb(self.pos(grid))
        grid = grid.mul(self.pass_html)

        grid = grid.add(plus)
        bbox = self.pos(html)

        return grid, bbox
