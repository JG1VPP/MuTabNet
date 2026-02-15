import torch
import torch.nn as nn
from positional_encodings import torch_encodings as pos

from mutab.block import Blocks, Linear
from mutab.utils import MODELS


@MODELS.register_module()
class TableCellUpdater(nn.Module):
    def __init__(
        self,
        d_input: int,
        d_model: int,
        num_emb: int,
        steps: int,
        PAD: int,
        EOS: int,
        **kwargs,
    ):
        super().__init__()

        # special tokens
        self.register_buffer("PAD", torch.as_tensor(PAD))
        self.register_buffer("EOS", torch.as_tensor(EOS))

        # embedding
        self.emb = nn.Embedding(num_emb, d_model, max_norm=1)
        self.pos = pos.PositionalEncoding1D(channels=d_model)

        # blocks
        self.net = Blocks(d_model=d_model, **kwargs)
        self.mix = Linear(d_input, d_model)
        self.out = Linear(d_model, num_emb)

        # steps
        self.steps = steps

    def forward(self, img, seq, aux, time):
        mask = self.detect(seq)
        temp = self.remask(seq, time=time)

        return self.handle(img, temp, aux, mask)

    def predict(self, img, seq, aux):
        mask = self.detect(seq)
        temp = self.PAD.expand_as(seq)

        for time in torch.linspace(1, 0, self.steps):
            edit = self.handle(img, temp, aux, mask)

            temp = self.unmask(temp, edit)
            temp = self.remask(temp, time)

        return temp

    def handle(self, img, seq, aux, mask):
        assert seq.ndim == 2
        assert aux.ndim == 3

        # embedding
        emb = self.emb(seq)
        pos = self.pos(emb)

        # perform inference
        hid = self.mix(aux.detach()).add(emb).add(pos)
        hid = self.net(x=hid, y=img, mask=mask.bool())

        return self.out(hid)

    def detect(self, seq):
        mask = seq.eq(self.EOS).cumsum(dim=1)
        mask = mask.unsqueeze(1).unsqueeze(2)

        return mask.logical_not()

    def remask(self, seq, time):
        rand = torch.rand(seq.shape, device=seq.device)
        return seq.masked_fill(rand.lt(time), self.PAD)

    def unmask(self, seq, edit):
        return seq.where(self.PAD.ne(seq), edit.argmax(dim=2))
