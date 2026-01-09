from typing import Any, List, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from mutab.model.factory import MODELS, build


class BN(nn.BatchNorm2d):
    def __init__(self, d: int, mom=0.1):
        super().__init__(d, momentum=mom)


class Conv(nn.Conv2d):
    def __init__(self, d: int, h: int, k: int):
        super().__init__(d, h, k, padding=k // 2, bias=False)


class ConvBn(nn.Sequential):
    def __init__(self, d: int, h: int, k: int, mom=0.1):
        super().__init__(Conv(d, h, k), BN(h, mom=mom))


class ConvBnReLU(nn.Sequential):
    def __init__(self, d: int, h: int, k: int, mom=0.1):
        super().__init__(ConvBn(d, h, k, mom=mom), nn.ReLU())


@MODELS.register_module()
class GCA(nn.Module):
    def __init__(self, d: int, ratio: float, heads: int):
        super().__init__()
        neck = int(ratio * d)
        assert d % heads == 0
        self.size = d // heads
        self.prob = nn.Softmax(dim=2)
        self.mask = nn.Conv2d(self.size, 1, 1)
        self.norm = nn.LayerNorm([neck, 1, 1])
        self.c1 = nn.Conv2d(d, neck, 1)
        self.c2 = nn.Conv2d(neck, d, 1)

    def forward(self, x):
        n, c, h, w = x.size()
        mask = self.mask(x.reshape(-1, self.size, h, w))
        mask = self.prob(mask.flatten(-2).unsqueeze(-1))
        y = x.reshape(-1, self.size, h * w).unsqueeze(1)
        y = torch.matmul(y, mask).reshape(n, c, 1, 1)
        return self.c2(F.relu(self.norm(self.c1(y)))).add(x)


class ResidualBlock(nn.Module):
    def __init__(self, d: int, h: int, gca: List[str] = [], **gcb):
        super().__init__()
        self.cv1 = nn.Sequential()
        self.cv1.append(ConvBn(d, h, 3, mom=0.9))
        self.cv1.append(nn.ReLU())
        self.cv1.append(ConvBn(h, h, 3, mom=0.9))
        self.cv1.extend(build(gcb, type=gc, d=h) for gc in gca)
        self.cv2 = ConvBn(d, h, 1) if d != h else nn.Identity()

    def forward(self, x):
        return F.relu(self.cv2(x).add(self.cv1(x)))


class ResidualGroup(nn.Sequential):
    def __init__(self, d: int, h: int, depth: int, **gcb):
        super().__init__()
        self.append(ResidualBlock(d, h, **gcb))
        self.extend(ResidualBlock(h, h) for _ in range(1, depth))


@MODELS.register_module()
class TableResNet(nn.Sequential):
    def __init__(
        self,
        dim: int,
        out: int,
        gcb1: Mapping[str, Any],
        gcb2: Mapping[str, Any],
        gcb3: Mapping[str, Any],
        gcb4: Mapping[str, Any],
    ):
        super().__init__()

        ch1 = out // 8
        ch2 = out // 4
        ch3 = out // 2

        # group1
        self.append(ConvBnReLU(dim, ch1, 3))
        self.append(ConvBnReLU(ch1, ch2, 3))

        # group2
        self.append(nn.MaxPool2d(2, ceil_mode=True))
        self.append(ResidualGroup(ch2, ch3, **gcb1))

        # group3
        self.append(nn.MaxPool2d(2, ceil_mode=True))
        self.append(ResidualGroup(ch3, ch3, **gcb2))

        # group4
        self.append(nn.MaxPool2d(2, ceil_mode=True))
        self.append(ResidualGroup(ch3, out, **gcb3))
        self.append(ResidualGroup(out, out, **gcb4))
