from mmengine import MODELS, OPTIMIZERS, build_from_cfg
from ranger.ranger2020 import Ranger

OPTIMIZERS.register_module(module=Ranger)


def build(cfg, registry=MODELS, **kwargs):
    return build_from_cfg(dict(**cfg, **kwargs), registry)
