import abc
import math
from functools import partial
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from positional_encodings import torch_encodings as pos
from rotary_embedding_torch import RotaryEmbedding

from mutab.models.factory import ATTENTIONS, DECODERS, build_attention


class PositionalEncodingAdd(pos.PositionalEncoding1D):
    def forward(self, x):
        return super().forward(x).add(x)


class Mask(nn.Module):
    def forward(self, x, mask):
        return x.where(mask, torch.finfo(x.dtype).min)


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
        q = self.q(q).view(len(q), *self.lhd).transpose(1, 2)
        k = self.k(k).view(len(k), *self.lhd).transpose(1, 2)
        v = self.v(v).view(len(v), *self.lhd).transpose(1, 2)
        x = self.attention(q, k, v, **kwargs).transpose(1, 2)
        return self.w(x.contiguous().flatten(start_dim=2))

    @property
    @abc.abstractmethod
    def causal(self) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def attention(self, q, k, v, **kwargs):
        raise NotImplementedError


@ATTENTIONS.register_module()
class GlobalAttention(Attention):
    def __init__(self, dropout: float, **kwargs):
        super().__init__(**kwargs)
        self.drop = nn.Dropout(dropout)
        self.mask = Mask()

    @property
    def causal(self):
        return False

    def attention(self, q, k, v, mask=None, **kwargs):
        p = q.matmul(k.mT.div(math.sqrt(v.size(-1))))
        p = p if mask is None else self.mask(p, mask)
        return self.drop(p.softmax(dim=-1)).matmul(v)


@ATTENTIONS.register_module()
class WindowAttention(GlobalAttention):
    def __init__(self, window: int, **kwargs):
        super().__init__(**kwargs)
        self.rotary = RotaryEmbedding(self.dim)
        self.window = window

    @property
    def causal(self):
        return True

    def attention(self, q, k, v, **kwargs):
        # buckets
        bq = self.bucket(q)
        bk = self.unfold(self.bucket(k))
        bv = self.unfold(self.bucket(v))

        # indices
        n = int(bq.shape[-3:-1].numel())
        i = torch.arange(n).to(q.device)
        i = self.bucket(i.unsqueeze(-1))
        j = self.unfold(i).mT

        # masking
        mask = i.ge(j).logical_and(j.ne(-1))

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


@ATTENTIONS.register_module()
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
        self.att1 = build_attention(att1, **kwargs)
        self.att2 = build_attention(att2, **kwargs)
        self.feed = FeedForward(**kwargs)

    def forward(self, kwargs):
        kwargs.update(**self.perform(**kwargs))
        return kwargs

    def perform(self, x, y, mask=None, **kwargs):
        x = x.add(self.att1(x, x, x, mask=mask))
        x = x.add(self.att2(x, y, y, mask=None))
        x = x.add(self.feed(x))
        return dict(x=x)


class Blocks(nn.Sequential):
    def __init__(self, blocks, **kwargs):
        block = lambda args: Block(**args, **kwargs)
        super().__init__(*tuple(map(block, blocks)))

    def forward(self, **kwargs):
        return super().forward(kwargs).get("x")


class Fetcher(nn.Module):
    def __init__(self, SOC: int, EOS: int, **kwargs):
        super().__init__()

        # special tokens
        self.register_buffer("SOC", torch.tensor(SOC))
        self.register_buffer("EOS", torch.tensor(EOS))

    def extract(self, x, mask, size):
        return F.pad(x[mask], pad=(0, 0, 0, size - sum(mask)))

    def forward(self, img, hid, seq):
        assert hid.ndim == 3
        assert seq.ndim == 2

        # masking
        soc = torch.isin(seq, self.SOC).unsqueeze(2)
        eos = torch.isin(seq, self.EOS).unsqueeze(2)

        # padding
        soc = soc.logical_and(eos.cumsum(dim=1).logical_not())
        pad = partial(self.extract, size=soc.sum(dim=1).max())

        # extract
        ext = torch.stack(list(map(pad, hid, soc.squeeze(2))))

        return hid, ext


class Decoder(nn.Module):
    def __init__(
        self,
        d_input: int,
        d_model: int,
        num_emb: int,
        max_len: int,
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
        self.emb = nn.Embedding(num_emb, d_model)
        self.pos = PositionalEncodingAdd(d_model)

        # blocks
        self.dec = Blocks(d_model=d_model, **kwargs)
        self.cat = Linear(d_input, d_model)
        self.out = Linear(d_model, num_emb)

        # prediction length
        self.max_len = max_len

    def predict(self, img, aux):
        seq = self.SOS.expand(len(img), 1)
        eos = self.EOS.expand(len(img), 1)
        for _ in range(self.max_len + 1):
            h, out = self(img, seq, aux, argmax=True)
            seq = torch.cat([seq[:, :1], out], dim=1)
            end = seq.eq(eos).sum(dim=1).bool().sum()
            if end.item() == len(img):
                break

        return h, out

    def forward(self, img, seq, aux, argmax=False):
        # alignment
        idx = torch.eq(seq, self.SEP).cumsum(dim=1).unsqueeze(-1)
        mat = torch.zeros(*seq.shape, aux.size(1)).to(aux.device)
        mat = mat.scatter_(-1, idx.clip_(max=aux.size(1) - 1), 1)
        mix = torch.cat([self.emb(seq), mat.matmul(aux)], dim=-1)

        # prediction
        hid = self.dec(x=self.pos(self.cat(mix)), y=img, mask=None)
        out = self.out(hid).argmax(-1) if argmax else self.out(hid)

        return hid, out


@DECODERS.register_module()
class TableDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        html_decoder,
        cell_decoder,
        html_fetcher,
        num_emb_html: int,
        num_emb_cell: int,
        max_len_html: int,
        max_len_cell: int,
        SOC_HTML: List[int],
        SOS_HTML: int,
        EOS_HTML: int,
        SOS_CELL: int,
        EOS_CELL: int,
        SEP_CELL: int,
        **kwargs,
    ):
        super().__init__()

        # parameters
        html_decoder.update(d_model=d_model)
        cell_decoder.update(d_model=d_model)

        # alphabet
        html_decoder.update(num_emb=num_emb_html)
        cell_decoder.update(num_emb=num_emb_cell)

        # capacity
        html_decoder.update(max_len=max_len_html)
        cell_decoder.update(max_len=max_len_cell)

        # special tokens
        html_decoder.update(SOS=SOS_HTML)
        html_decoder.update(EOS=EOS_HTML)
        html_decoder.update(SEP=EOS_HTML)

        cell_decoder.update(SOS=SOS_CELL)
        cell_decoder.update(EOS=EOS_CELL)
        cell_decoder.update(SEP=SEP_CELL)

        html_fetcher.update(SOC=SOC_HTML)
        html_fetcher.update(EOS=EOS_HTML)

        # input channels
        html_decoder.update(d_input=d_model + 2)
        cell_decoder.update(d_input=d_model * 2)

        # other parameters
        html_decoder.update(**kwargs)
        cell_decoder.update(**kwargs)

        # en/decoders
        self.html = Decoder(**html_decoder)
        self.cell = Decoder(**cell_decoder)
        self.grid = Fetcher(**html_fetcher)

        # bbox
        self.bbox = Linear(d_model, 4, act=nn.Sigmoid)

        # LtoR or RtoL
        self.register_buffer("LtoR", torch.eye(2)[0])
        self.register_buffer("RtoL", torch.eye(2)[1])

    def forward(self, img, html, back, cell, **kwargs):
        # ground truth
        html = html.to(img.device)
        back = back.to(img.device)
        cell = cell.to(img.device)

        # remove [EOS]
        s_html = html[:, :-1]
        e_back = back[:, :-1]
        s_cell = cell[:, :-1]

        # remove [SOS]
        e_html = html[:, 1::]

        # LtoR or RtoL
        h_LtoR = self.LtoR.expand(len(img), 1, 2)
        h_RtoL = self.RtoL.expand(len(img), 1, 2)

        # structure prediction
        h_html, o_html = self.html(img, s_html, h_LtoR)
        h_back, o_back = self.html(img, e_back, h_RtoL)

        # character prediction
        h_html, h_grid = self.grid(img, h_html, e_html)
        h_cell, o_cell = self.cell(img, s_cell, h_grid)

        return dict(
            html=o_html,
            back=o_back,
            cell=o_cell,
            bbox=self.bbox(h_html),
        )

    def predict(self, img):
        # LtoR
        h_LtoR = self.LtoR.expand(len(img), 1, 2)

        # structure prediction
        h_html, o_html = self.html.predict(img, h_LtoR)

        # character prediction
        h_html, h_grid = self.grid(img, h_html, o_html)
        h_cell, o_cell = self.cell.predict(img, h_grid)

        return dict(html=o_html, cell=o_cell, bbox=self.bbox(h_html))
