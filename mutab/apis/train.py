import os
from datetime import datetime as dt

from mmcv import Config, mkdir_or_exist
from mmcv.runner import get_dist_info
from mmdet.apis import train_detector
from mmdet.utils import get_device

from mutab.datasets import build_dataset
from mutab.models import build_detector
from mutab.utils import collect_env, get_logger, pretty_env


def train(cfg: Config, cfg_file: str):
    mkdir_or_exist(cfg.work_dir)
    _, devices = get_dist_info()
    cfg.gpu_ids = range(devices)
    cfg.device = get_device()

    # prepare log
    time = dt.now().strftime("%Y%m%d_%H%M%S")
    log = os.path.join(cfg.work_dir, "{}.log".format(time))
    log = get_logger(log_file=log, log_level=cfg.log_level)
    metas = dict(env=collect_env(), config=cfg.pretty_text)

    # dump environmental information
    log.info(pretty_env(bar="-" * 64))
    log.info("\n{}".format(cfg.pretty_text))

    # build model and dataset
    model = build_detector(cfg.model)
    dataset = build_dataset(cfg.data.train)

    # dump configuration
    os.environ.update(LOCAL_RANK=os.getenv("LOCAL_RANK", "0"))
    cfg.dump(os.path.join(cfg.work_dir, os.path.basename(cfg_file)))

    # start training
    cfg.checkpoint_config.meta = dict(env=collect_env(), CLASSES=int(dataset.CLASSES))
    train_detector(model, dataset, cfg, devices > 1, True, timestamp=time, meta=metas)
