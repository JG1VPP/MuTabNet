import torch
import torch.nn as nn
import torch.nn.functional as F

from mutab.block import Blocks
from mutab.utils import MODELS


@MODELS.register_module()
class TableCellFetcher(nn.Module):
    def __init__(self, SOC: int, EOS: int, **kwargs):
        super().__init__()

        # special tokens
        self.register_buffer("SOC", torch.as_tensor(SOC))
        self.register_buffer("EOS", torch.as_tensor(EOS))

        # blocks
        self.md = Blocks(**kwargs)

    def forward(self, img, hid, seq):
        assert hid.ndim == 3
        assert seq.ndim == 2

        # detect full cells
        soc = torch.isin(seq, self.SOC).unsqueeze(2)
        eos = torch.isin(seq, self.EOS).unsqueeze(2)

        # mask excess cells
        soc.logical_and_(eos.cumsum(dim=1).logical_not())

        # remove pad tokens
        ext = self.extract(hid, mask=soc.squeeze(2))
        run = self.extract(soc, mask=soc.squeeze(2))

        # perform inference
        ext = self.md(x=ext, y=img, mask=run.unsqueeze(1).mT)
        hid = hid.masked_scatter(soc, ext.masked_select(run))

        return hid, ext

    def extract(self, data, mask, pad: int = 0):
        def move(item, mask):
            pad = (0, 0, 0, size - sum(mask))
            return F.pad(item[mask], pad=pad)

        size = mask.count_nonzero(dim=-1).add(pad).max()
        return torch.stack(tuple(map(move, data, mask)))
