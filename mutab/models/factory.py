from mmcv.utils import build_from_cfg
from mmdet.models.builder import MODELS


def build(cfg, registry=MODELS, **kwargs):
    return build_from_cfg(dict(**cfg, **kwargs), registry)
