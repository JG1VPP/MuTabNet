from mmengine import MODELS, build_from_cfg


def build(cfg, registry=MODELS, **kwargs):
    return build_from_cfg(dict(**cfg, **kwargs), registry)
