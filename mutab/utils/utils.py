import mmcv.utils as utils


def get_logger(**kwargs):
    return utils.get_logger("mmdet", **kwargs)


def collect_env():
    return dict(**utils.collect_env(), commit=utils.get_git_hash())


def pretty_env(bar: str):
    contents = list(f"{k}: {v}" for k, v in collect_env().items())
    return "\n".join(["", bar] + contents + [bar, ""])
