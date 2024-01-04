from mmcv.utils import Registry, build_from_cfg
from mmdet.models.builder import BACKBONES, DETECTORS, LOSSES

HANDLERS = Registry("handler")
ENCODERS = Registry("encoder")
DECODERS = Registry("decoder")
ATTENTIONS = Registry("attentions")


def build_from_dict(cfg, registry, **kwargs):
    return build_from_cfg(dict(**cfg, **kwargs), registry)


def build_detector(cfg, **kwargs):
    return build_from_dict(cfg, DETECTORS, **kwargs)


def build_backbone(cfg, **kwargs):
    return build_from_dict(cfg, BACKBONES, **kwargs)


def build_encoder(cfg, **kwargs):
    return build_from_dict(cfg, ENCODERS, **kwargs)


def build_decoder(cfg, **kwargs):
    return build_from_dict(cfg, DECODERS, **kwargs)


def build_handler(cfg, **kwargs):
    return build_from_dict(cfg, HANDLERS, **kwargs)


def build_loss(cfg, **kwargs):
    return build_from_dict(cfg, LOSSES, **kwargs)


def build_attention(cfg, **kwargs):
    return build_from_dict(cfg, ATTENTIONS, **kwargs)
