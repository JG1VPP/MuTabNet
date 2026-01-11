from itertools import count, product
from pathlib import Path

import torch
import torch.nn as nn
from more_itertools import collapse, split_at

from mutab.utils import MODELS


@MODELS.register_module()
class TableLexicon(nn.Module):
    def __init__(self, load: str, **kwargs):
        super().__init__()

        self.fwd = dict()
        self.bwd = dict()

        # special tokens
        self.pad = self.special("PAD")
        self.sos = self.special("SOS")
        self.eos = self.special("EOS")
        self.sep = self.special("SEP")
        self.ukn = self.special("UKN")

        # load
        fwd, bwd = self.parse(load)
        assert len(fwd) == len(bwd)

    def special(self, name: str):
        token = "<{}>".format(name)

        tensor = self.get_tensor(len(self.fwd))
        self.register_buffer(str(name), tensor)

        self.fwd.update({token: tensor.item()})
        self.bwd.update({tensor.item(): token})

        return token

    def parse(self, path: str):
        skip = len(self.fwd)

        voc = Path(path).read_text()
        voc = list(voc.splitlines())

        self.fwd.update(zip(voc, count(skip)))
        self.bwd.update(zip(count(skip), voc))

        return self.fwd, self.bwd

    @property
    def device(self):
        return self.PAD.device

    @property
    def num_class(self):
        return len(self.fwd)

    def forward(self, seq):
        seq = map(self.get_number, seq)
        seq = map(self.get_tensor, seq)

        return torch.stack(tuple(seq))

    def reverse(self, seq):
        eos = seq.eq(self.EOS).cumsum(dim=0).logical_not()
        seq = map(self.get_string, seq.masked_select(eos))

        return tuple(seq)

    def get_tensor(self, value):
        return torch.as_tensor(value)

    def get_number(self, token):
        return self.fwd.get(token, self.UKN.item())

    def get_string(self, token):
        return self.bwd.get(int(token))

    def enclose(self, seq: list[str]):
        return (self.sos, *seq, self.eos)

    def flatten(self, seq: list[list[str]]):
        return tuple(collapse(product(seq, [[self.sep]])))

    def resolve(self, seq: list[str]):
        return tuple(split_at(seq, lambda v: v == self.sep))
