from inspect import isclass

from mmcv.runner.optimizer.builder import OPTIMIZERS
from ranger.ranger2020 import Ranger
from torch import optim
from torch.optim import Optimizer


def register_torch_optimizers():
    for name in dir(optim):
        if name.startswith("__"):
            continue
        _optim = getattr(optim, name)
        if isclass(_optim) and issubclass(_optim, Optimizer):
            if name not in OPTIMIZERS.module_dict.keys():
                OPTIMIZERS.register_module()(_optim)

    if isclass(Ranger) and issubclass(Ranger, Optimizer):
        OPTIMIZERS.register_module()(Ranger)


register_torch_optimizers()
