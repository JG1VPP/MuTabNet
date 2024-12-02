import abc
import math
from operator import itemgetter
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from local_attention import LocalAttention
from positional_encodings import torch_encodings as pos

from mutab.models.factory import ATTENTIONS, DECODERS, build_attention


class PositionalEncodingAdd(pos.PositionalEncoding1D):
    def forward(self, x):
        return super().forward(x).add(x)


class Linear(nn.Sequential):
    def __init__(self, d: int, h: int, act=nn.Identity):
        super().__init__(nn.LayerNorm(d), nn.Linear(d, h), act())


class Linears(nn.Sequential):
    def __init__(self, d: int, h: int, act=nn.ReLU):
        super().__init__(Linear(d, d, act), Linear(d, h))


class Attention(nn.Module, abc.ABC):
    def __init__(self, heads: int, d_model: int, **kwargs):
        super().__init__()
        assert d_model % heads == 0
        self.dim = int(d_model / heads)
        self.size = heads, self.dim
        self.fc_q = Linear(d_model, d_model)
        self.fc_k = Linear(d_model, d_model)
        self.fc_v = Linear(d_model, d_model)
        self.fc_x = Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        q = self.fc_q(q).view(len(q), -1, *self.size).swapaxes(1, 2)
        k = self.fc_k(k).view(len(k), -1, *self.size).swapaxes(1, 2)
        v = self.fc_v(v).view(len(v), -1, *self.size).swapaxes(1, 2)
        x = self.attention(q, k, v, mask=mask).swapaxes(1, 2)
        return self.fc_x(x.contiguous().flatten(start_dim=2))

    @abc.abstractmethod
    def attention(self, q, k, v, mask):
        raise NotImplementedError

    @abc.abstractmethod
    def causality(self, ksize: int):
        raise NotImplementedError


@ATTENTIONS.register_module()
class WindowAttention(Attention):
    def __init__(self, dropout: float, window: int, **kwargs):
        super().__init__(**kwargs)
        self.local_att = LocalAttention(
            dim=self.dim,
            window_size=window,
            causal=True,
            look_forward=0,
            look_backward=1,
            dropout=dropout,
            autopad=True,
            exact_windowsize=False,
        )

    def attention(self, q, k, v, mask):
        return self.local_att(q, k, v, mask=mask)

    def causality(self, ksize: int):
        return None


@ATTENTIONS.register_module()
class GlobalAttention(Attention):
    def __init__(self, dropout: float, **kwargs):
        super().__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    def attention(self, q, k, v, mask):
        prob = torch.matmul(q, k.swapaxes(-2, -1) / math.sqrt(v.shape[-1]))
        mask = torch.ones_like(prob) if mask is None else mask.to(q.device)
        prob = prob.masked_fill(mask.logical_not(), torch.finfo(q.dtype).min)
        return self.dropout(prob.softmax(dim=-1)).matmul(v).reshape(*q.shape)

    def causality(self, ksize: int):
        return torch.ones(ksize, ksize).tril_().expand(1, 1, ksize, ksize)


class Block(nn.Module):
    def __init__(self, type1, type2, d_model: int, **kwargs):
        super().__init__()
        self.att1 = build_attention(kwargs, type=type1, d_model=d_model)
        self.att2 = build_attention(kwargs, type=type2, d_model=d_model)
        self.norm = Linears(d_model, d_model)

    def forward(self, features):
        x, y = itemgetter("x", "y")(features)
        mask = self.att1.causality(x.size(1))
        x = x + self.att1(x, x, x, mask=mask)
        x = x + self.att2(x, y, y)
        x = x + self.norm(x)
        return dict(x=x, y=y)


class Blocks(nn.Sequential):
    def __init__(self, depth: int, **decoder):
        block = lambda n: Block(index=n, **decoder)
        super().__init__(*map(block, range(depth)))


class Decoder(nn.Module):
    def __init__(
        self,
        depth: int,
        d_input: int,
        d_model: int,
        num_emb: int,
        max_len: int,
        SOS: int,
        EOS: int,
        SEP: int,
        **decoder,
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
        self.dec = Blocks(depth, d_model=d_model, **decoder)
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
        hid = self.dec(dict(x=self.pos(self.cat(mix)), y=img)).get("x")
        out = self.out(hid).argmax(dim=-1) if argmax else self.out(hid)

        return hid, out


@DECODERS.register_module()
class TableDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        html_decoder,
        cell_decoder,
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

        # en/decoders
        self.html = Decoder(d_input=d_model + 2, **html_decoder)
        self.cell = Decoder(d_input=d_model * 2, **cell_decoder)

        # bbox
        self.bbox = Linear(d_model, 4, nn.Sigmoid)

        # mask
        self.register_buffer("SOC_HTML", torch.tensor(SOC_HTML))

        # LtoR or RtoL
        self.register_buffer("LtoR", torch.eye(2)[0])
        self.register_buffer("RtoL", torch.eye(2)[1])

    def forward(self, img, html, back, cell, **kwargs):
        # ground truth
        html = html.to(img.device)
        back = back.to(img.device)
        cell = cell.to(img.device)

        # LtoR or RtoL
        LtoR = self.LtoR.expand(len(img), 1, 2)
        RtoL = self.RtoL.expand(len(img), 1, 2)

        # structure prediction
        html_hid, html_out = self.html(img, html[:, :-1], LtoR)
        back_hid, back_out = self.html(img, back[:, :-1], RtoL)

        # <TD></TD> extraction
        grid = self.grid(html_hid, html[:, 1:])

        # character prediction
        cell_hid, cell_out = self.cell(img, cell[:, :-1], grid)

        return dict(
            html=html_out,
            back=back_out,
            cell=cell_out,
            bbox=self.bbox(html_hid),
        )

    def predict(self, img):
        # LtoR
        LtoR = self.LtoR.expand(len(img), 1, 2)

        # structure prediction
        html_hid, html_out = self.html.predict(img, LtoR)

        # cell bbox prediction
        _, cell_out = self.cell.predict(img, self.grid(html_hid, html_out))
        return dict(html=html_out, cell=cell_out, bbox=self.bbox(html_hid))

    def grid(self, x, text):
        pad = lambda x, mask: F.pad(x[mask], (0, 0, 0, len(x) - len(x[mask])))
        return torch.stack(list(map(pad, x, torch.isin(text, self.SOC_HTML))))
