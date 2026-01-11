import abc
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from rotary_embedding_torch import RotaryEmbedding

from mutab.utils import MODELS, build


class Mask(nn.Module):
    def forward(self, x, mask):
        return x.where(mask.to(x.device), torch.finfo(x.dtype).min)


class Linear(nn.Sequential):
    def __init__(self, d: int, h: int, *, act=nn.Identity):
        super().__init__(nn.LayerNorm(d), nn.Linear(d, h), act())


class Attention(nn.Module, abc.ABC):
    def __init__(self, heads: int, d_model: int, **kwargs):
        super().__init__()
        assert d_model % heads == 0
        self.dim = int(d_model // heads)
        self.lhd = (-1, heads, self.dim)
        self.q = Linear(d_model, d_model)
        self.k = Linear(d_model, d_model)
        self.v = Linear(d_model, d_model)
        self.w = Linear(d_model, d_model)

    def forward(self, q, k, v, **kwargs):
        x = self.q(q).view(len(q), *self.lhd).swapaxes(1, 2)
        k = self.k(k).view(len(k), *self.lhd).swapaxes(1, 2)
        v = self.v(v).view(len(v), *self.lhd).swapaxes(1, 2)
        x = self.attention(x, k, v, **kwargs).swapaxes(1, 2)
        return self.w(x.contiguous().flatten(-2)).view_as(q)

    @property
    @abc.abstractmethod
    def causal(self) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def attention(self, q, k, v, **kwargs):
        raise NotImplementedError


@MODELS.register_module()
class GlobalAttention(Attention):
    def __init__(self, dropout: float, **kwargs):
        super().__init__(**kwargs)
        self.drop = nn.Dropout(dropout)
        self.mask = Mask()
        self.hook = lambda p: p

    @property
    def causal(self):
        return False

    def weight(self, p, q):
        return self.hook(p.softmax(dim=-1))

    def attention(self, q, k, v, mask=None, **kwargs):
        p = q.matmul(k.mT.div(math.sqrt(v.size(-1))))
        p = p if mask is None else self.mask(p, mask)
        return self.drop(self.weight(p, q)).matmul(v)


@MODELS.register_module()
class CausalAttention(GlobalAttention):
    @property
    def causal(self):
        return True

    def attention(self, q, k, v, **kwargs):
        mask = torch.ones(q.size(-2), k.size(-2), dtype=bool)
        return super().attention(q, k, v=v, mask=mask.tril())


@MODELS.register_module()
class WindowAttention(GlobalAttention):
    def __init__(self, window: int, **kwargs):
        super().__init__(**kwargs)
        self.rotary = RotaryEmbedding(self.dim)
        self.window = window

    @property
    def causal(self):
        return True

    def attention(self, q, k, v, cell=None, **kwargs):
        # buckets
        bq = self.bucket(q)
        bk = self.unfold(self.bucket(k))
        bv = self.unfold(self.bucket(v))

        # indices
        n = int(bq.shape[-3:-1].numel())
        i = torch.arange(n).to(q.device)
        i = self.bucket(i.unsqueeze(-1))
        j = self.unfold(i).mT

        # regions
        if cell is not None:
            req = self.bucket(cell)
            rek = self.unfold(req).mT
        else:
            req = torch.ones(1).to(i)
            rek = torch.ones(1).to(j)

        # masking
        mask = i.ge(j).logical_and(j.ne(-1))
        mask = mask.logical_and(req.eq(rek))

        # rotary embedding
        bq = self.rotary.rotate_queries_or_keys(bq)
        bk = self.rotary.rotate_queries_or_keys(bk)

        # global attention
        out = super().attention(q=bq, k=bk, v=bv, mask=mask)
        return out.flatten(-3, -2).narrow(-2, 0, q.size(-2))

    def bucket(self, x):
        n = self.window * math.ceil(x.size(-2) / self.window)
        x = F.pad(x, pad=(0, 0, 0, n - x.size(-2)), value=-1)
        x = torch.stack(x.split(self.window, dim=-2), dim=-3)
        return x

    def unfold(self, x):
        pad = F.pad(x, pad=(0, 0, 0, 0, 1, 0), value=-1)
        pad = pad.narrow(-3, start=0, length=x.size(-3))
        return torch.cat([pad, x], dim=-2)


@MODELS.register_module()
class AbsentAttention(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, q, k, v, **kwargs):
        return torch.zeros_like(q)


class FeedForward(nn.Sequential):
    def __init__(self, d_model: int, **kwargs):
        super().__init__()
        self.append(Linear(d_model, d_model, act=nn.ReLU))
        self.append(Linear(d_model, d_model, act=nn.Identity))


class Block(nn.Module):
    def __init__(self, att1, att2, **kwargs):
        super().__init__()
        self.att1 = build(att1, **kwargs)
        self.att2 = build(att2, **kwargs)
        self.feed = FeedForward(**kwargs)

    def forward(self, kwargs):
        kwargs.update(**self.perform(**kwargs))
        return kwargs

    def perform(self, x, y, cell=None, mask=None, **kwargs):
        x = x.add(self.att1(x, x, x, cell=cell, mask=mask))
        x = x.add(self.att2(x, y, y, cell=None, mask=None))
        x = x.add(self.feed(x))
        return dict(x=x)


class Blocks(nn.Sequential):
    def __init__(self, blocks, **kwargs):
        block = lambda args: Block(**args, **kwargs)
        super().__init__(*tuple(map(block, blocks)))

    def forward(self, **kwargs):
        return super().forward(kwargs).get("x")
