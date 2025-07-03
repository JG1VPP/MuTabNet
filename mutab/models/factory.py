from functools import partial

from mmcv.utils import Registry, build_from_cfg
from mmdet.models.builder import BACKBONES, DETECTORS, LOSSES

DECODERS = Registry("decoder")
ENCODERS = Registry("encoder")
HANDLERS = Registry("handler")
NETWORKS = Registry("network")
ATTENTIONS = Registry("attention")
GC_MODULES = Registry("gc-module")


def build_from_dict(cfg, registry, **kwargs):
    return build_from_cfg(dict(**cfg, **kwargs), registry)


build_loss = partial(build_from_dict, registry=LOSSES)
build_detector = partial(build_from_dict, registry=DETECTORS)
build_backbone = partial(build_from_dict, registry=BACKBONES)
build_decoder = partial(build_from_dict, registry=DECODERS)
build_encoder = partial(build_from_dict, registry=ENCODERS)
build_handler = partial(build_from_dict, registry=HANDLERS)
build_network = partial(build_from_dict, registry=NETWORKS)
build_attention = partial(build_from_dict, registry=ATTENTIONS)
build_gc_module = partial(build_from_dict, registry=GC_MODULES)
