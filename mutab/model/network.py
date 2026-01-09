import sys
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from positional_encodings import torch_encodings as pos

from mutab.block import Blocks, Linear
from mutab.model.factory import MODELS


class Network(nn.Module):
    def predict(self, *args, **kwargs):
        *hid, cls = self(*args, **kwargs)
        return (*hid, cls.argmax(dim=-1))

    def extract(self, data, mask, pad: int = 0):
        def move(item, mask):
            pad = (0, 0, 0, size - sum(mask))
            return F.pad(item[mask], pad=pad)

        size = mask.count_nonzero(dim=-1).add(pad).max()
        return torch.stack(tuple(map(move, data, mask)))


@MODELS.register_module()
class Fetcher(Network):
    def __init__(self, SOC: int, EOS: int, **kwargs):
        super().__init__()

        # special tokens
        self.register_buffer("SOC", torch.tensor(SOC))
        self.register_buffer("EOS", torch.tensor(EOS))

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


@MODELS.register_module()
class Decoder(Network):
    def __init__(
        self,
        d_input: int,
        d_model: int,
        num_emb: int,
        SOS: int,
        EOS: int,
        SEP: int,
        **kwargs,
    ):
        super().__init__()

        # special tokens
        self.register_buffer("SOS", torch.tensor(SOS))
        self.register_buffer("EOS", torch.tensor(EOS))
        self.register_buffer("SEP", torch.tensor(SEP))

        # embedding
        self.emb = nn.Embedding(num_emb, d_model, max_norm=1)
        self.pos = pos.PositionalEncoding1D(channels=d_model)

        # blocks
        self.dec = Blocks(d_model=d_model, **kwargs)
        self.cat = Linear(d_input, d_model)
        self.out = Linear(d_model, num_emb)

        # other parameters
        self.set_parameters_or_default(**kwargs)

    def set_parameters_or_default(
        self,
        max_len: Optional[int] = None,
        max_spd: Optional[int] = None,
        **kwargs,
    ):
        self.MAX = max_len or sys.maxsize
        self.SPD = max_spd or sys.maxsize

    def forward(self, img, seq, aux):
        assert seq.ndim == 2
        assert aux.ndim == 3

        # max index of [SEP] token
        mux = int(aux.size(1) - 1)

        # detect token occurrences
        sep = seq.eq(self.SEP).unsqueeze(2)

        # indices inside each cell
        pos = torch.ones_like(sep).cumsum(dim=1).sub_(1)
        pos = pos.sub(pos.mul(sep).cummax(dim=1).values)

        # identify important cells
        idx = sep.cumsum(dim=1).clip(max=mux)

        # assign features to cells
        mat = torch.zeros(*seq.shape, mux + 1).to(aux)
        mat.scatter_(dim=seq.ndim, index=idx, value=1)

        # with positional encoding
        mix = self.cat(torch.dstack([self.emb(seq), mat.bmm(aux)]))
        pos = self.pos(mix).gather(dim=1, index=pos.expand_as(mix))

        # decode cells in parallel
        hid = self.dec(x=mix.add(pos), y=img, cell=idx.unsqueeze(1))

        return hid, self.out(hid)

    def predict(self, img, aux):
        yet = True

        # initial batch
        sep = aux.any(dim=2)
        sos = self.SOS.expand(len(aux), 1)
        eos = self.EOS.expand(len(aux), 1)
        sep = self.SEP.where(sep, self.EOS)
        seq = torch.hstack([sos, sep, eos])

        # sequential inference
        while yet and seq.size(1) <= self.MAX:
            seq, yet = self.enlarge(img, seq, aux)

        return super().predict(img, seq, aux)

    def enlarge(self, img, seq, aux):
        hid, out = super().predict(img, seq, aux)

        # detect new tokens
        mask = seq.roll(-1, 1).eq(self.SEP)
        mask.logical_and_(out.ne(self.SEP))
        mask.logical_and_(out.ne(self.EOS).cumprod(dim=1))
        mask.logical_and_(seq.ne(self.EOS).cumprod(dim=1))
        mask.logical_and_(mask.cumsum(dim=1).le(self.SPD))

        # remove old tokens
        out.mul_(mask.to(out))

        # aggregate inserts
        old = F.pad(mask, pad=(1, 0)).cumsum(dim=1)[:, :-1]
        new = F.pad(mask, pad=(0, 1)).cumsum(dim=1)[:, :-1]

        # calculate indices
        old.add_(torch.arange(old.size(1)).to(seq))
        new.add_(torch.arange(new.size(1)).to(seq))

        # allocate sequence
        ret = mask.cumsum(dim=1).max().add(old.size(1))
        ret = torch.zeros(len(seq), ret.item()).to(seq)

        # limit index range
        old.clip_(max=ret.size(1) - 1)
        new.clip_(max=ret.size(1) - 1)

        # combine sequences
        ret.scatter_add_(dim=1, index=old, src=seq)
        ret.scatter_add_(dim=1, index=new, src=out)

        return ret, mask.any().item()


@MODELS.register_module()
class Locator(Network):
    def __init__(self, d_model: int, pass_html: bool, **kwargs):
        super().__init__()
        self.pos = Linear(d_model, 4, act=nn.Sigmoid)
        self.emb = Linear(4, d_model, act=nn.Sigmoid)
        self.pass_html = int(pass_html)

    def forward(self, img, html, grid):
        plus = self.emb(self.pos(grid))
        grid = grid.mul(self.pass_html)
        grid = grid.add(plus)
        bbox = self.pos(html)
        return grid, bbox
