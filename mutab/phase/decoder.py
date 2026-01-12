import sys
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from positional_encodings import torch_encodings as pos

from mutab.block import Blocks, Linear
from mutab.utils import MODELS


@MODELS.register_module()
class TableCellDecoder(nn.Module):
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
        self.register_buffer("SOS", torch.as_tensor(SOS))
        self.register_buffer("EOS", torch.as_tensor(EOS))
        self.register_buffer("SEP", torch.as_tensor(SEP))

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

        return self.resolve(img, seq, aux)

    def enlarge(self, img, seq, aux):
        hid, out = self.resolve(img, seq, aux)

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

    def resolve(self, *args, **kwargs):
        *hid, cls = self(*args, **kwargs)
        return (*hid, cls.argmax(dim=-1))
