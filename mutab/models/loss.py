import torch
import torch.nn as nn

from mutab.models.factory import LOSSES


@LOSSES.register_module()
class CELoss(nn.Module):
    def __init__(self, key: str, ignore_index: int):
        super().__init__()

        # keys
        self.key = key
        self.label = "loss_ce_{}".format(key)

        # loss
        self.loss = self.build_loss(ignore_index)

    def build_loss(self, ignore_index):
        return nn.CrossEntropyLoss(ignore_index=ignore_index)

    def format(self, outputs, targets):
        # outputs [N, C, L]
        # targets [N, L]
        logit = outputs[self.key].mT
        label = targets[self.key][:, 1:]
        return logit, label.to(logit.device)

    def forward(self, outputs, targets, img_metas=None):
        logit, label = self.format(outputs, targets)
        return {self.label: self.loss(logit, label)}


@LOSSES.register_module()
class KLLoss(nn.Module):
    def __init__(self, key: str, rev: str, ignore_index: int):
        super().__init__()

        # keys
        self.key = key
        self.rev = rev

        # labels
        self.loss_key = f"loss_kl_{key}"
        self.loss_rev = f"loss_kl_{rev}"

        # prob
        self.sm_p = nn.Softmax(dim=2)
        self.sm_q = nn.LogSoftmax(dim=2)

        # loss
        self.loss = self.build_loss("sum")

        # <PAD>
        pad = torch.tensor(ignore_index)
        self.register_buffer("PAD", pad)

    def build_loss(self, reduction):
        return nn.KLDivLoss(reduction=reduction)

    def format(self, outputs, targets):
        # outputs [N, L, C]
        logit_f = outputs[self.key][:, :-1]
        logit_b = outputs[self.rev][:, :-1].fliplr()

        # detect <PAD>
        text = targets[self.key][:, 1:-1].unsqueeze(-1)
        mask = ~torch.isin(text.to(self.PAD), self.PAD)

        # P: target
        p_f = self.sm_p(logit_b.mul(mask)).detach()
        p_b = self.sm_p(logit_f.mul(mask)).detach()

        # Q: output
        q_f = self.sm_q(logit_f.mul(mask))
        q_b = self.sm_q(logit_b.mul(mask))

        return (q_f, p_f), (q_b, p_b), mask

    def forward(self, outputs, targets, img_metas=None):
        qp_f, qp_b, mask = self.format(outputs, targets)
        kl_f = self.loss(*qp_f).div(mask.sum().clamp(1))
        kl_b = self.loss(*qp_b).div(mask.sum().clamp(1))
        return {self.loss_key: kl_f, self.loss_rev: kl_b}


@LOSSES.register_module()
class BBLoss(nn.Module):
    def __init__(self, ignore_index: str):
        super().__init__()

        # loss
        self.loss = self.build_loss("sum")

        # <PAD>
        pad = torch.tensor(ignore_index)
        self.register_buffer("PAD", pad)

    def build_loss(self, reduction):
        return nn.L1Loss(reduction=reduction)

    def format(self, outputs, targets):
        # outputs [N, L, 4]
        pred = outputs["bbox"]

        # targets [N, L, 4]
        bbox = targets["bbox"][:, 1:].to(pred.device)

        # structural tokens
        html = targets["html"][:, 1:].to(pred.device)

        # detect <PAD>
        mask = ~torch.eq(html, self.PAD).unsqueeze(-1)

        # remove <PAD>
        pred = pred.masked_select(mask)
        bbox = bbox.masked_select(mask)

        assert pred.dim() == 1
        assert bbox.dim() == 1

        # samples
        pair_h = pred[0::2], bbox[0::2]
        pair_v = pred[1::2], bbox[1::2]

        return pair_h, pair_v, mask

    def forward(self, outputs, targets, img_metas=None):
        pair_h, pair_v, mask = self.format(outputs, targets)
        loss_h = self.loss(*pair_h).div(mask.sum().clamp(1))
        loss_v = self.loss(*pair_v).div(mask.sum().clamp(1))
        return dict(loss_h=loss_h, loss_v=loss_v)
