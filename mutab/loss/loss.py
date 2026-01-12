from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from mutab.utils import MODELS, build


class Loss(nn.Module, ABC):
    label = "loss_{}"

    def __init__(self, key: str, **kwargs):
        super().__init__()

        # keys
        self.key = key
        self.loss = self.function(**kwargs)
        self.label = self.label.format(key)

    @abstractmethod
    def function(self, **kwargs):
        pass

    def format(self, outputs, targets):
        pred = outputs[self.key]
        true = targets[self.key]

        return pred, true

    def forward(self, outputs, targets):
        inputs = self.format(outputs, targets)
        return {self.label: self.loss(*inputs)}


@MODELS.register_module()
class CELoss(Loss):
    label = "loss_ce_{}"

    def function(self, ignore: int, **kwargs):
        return nn.CrossEntropyLoss(ignore_index=ignore)

    def format(self, outputs, targets):
        # outputs [N, C, L]
        # targets [N, L]
        logit = outputs[self.key].mT
        label = targets[self.key][:, 1:]

        return logit, label


@MODELS.register_module()
class KLLoss(Loss):
    label = "loss_kl_{}"

    def function(self, ignore: int, rev: str, **kwargs):
        # key
        self.rev = rev

        # PAD
        pad = torch.tensor(ignore).int()
        self.register_buffer("PAD", pad)

        # prob
        self.p = nn.Softmax(dim=2)
        self.q = nn.LogSoftmax(dim=2)

        # loss
        return nn.KLDivLoss(reduction="sum")

    def format(self, outputs, targets):
        # outputs [N, L, C]
        logit_f = outputs[self.key][:, :-1]
        logit_b = outputs[self.rev][:, :-1].fliplr()

        # detect PAD
        text = targets[self.key][:, 1:-1]
        mask = text.ne(self.PAD).unsqueeze(-1)

        # P: target
        # Q: output
        p = self.p(logit_b.mul(mask)).detach()
        q = self.q(logit_f.mul(mask))

        return (q, p), mask.sum()

    def forward(self, outputs, targets):
        qp, denom = self.format(outputs, targets)
        loss = self.loss(*qp).div(denom.clamp(1))

        return {self.label: loss}


@MODELS.register_module()
class BBLoss(Loss):
    def function(self, ignore: int, cls: str, **kwargs):
        # key
        self.cls = cls

        # PAD
        pad = torch.tensor(ignore).int()
        self.register_buffer("PAD", pad)

        # MAE
        return nn.L1Loss()

    def format(self, outputs, targets):
        # outputs [N, L, 4]
        pred = outputs[self.key]

        # targets [N, L, 4]
        bbox = targets[self.key][:, 1:]

        # detect PAD
        mask = targets[self.cls][:, 1:].ne(self.PAD)

        # remove PAD
        pred = pred.masked_select(mask.unsqueeze(2))
        bbox = bbox.masked_select(mask.unsqueeze(2))

        return pred, bbox


@MODELS.register_module()
class Nested(Loss):
    def function(self, **kwargs):
        return build(kwargs["loss"])
