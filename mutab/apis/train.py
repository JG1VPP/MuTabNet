import os
from datetime import datetime
from pathlib import Path

from mmcv import Config, mkdir_or_exist
from mmcv.runner import get_dist_info
from mmdet.apis import train_detector
from mmdet.utils import get_device

from mutab import utils
from mutab.datasets import build_dataset
from mutab.models import build_detector


def train(cfg: Config, cfg_file: str):
    mkdir_or_exist(cfg.work_dir)
    _, devices = get_dist_info()
    cfg.gpu_ids = range(devices)
    cfg.device = get_device()

    # prepare log
    time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log = Path(cfg.work_dir).joinpath(time).with_suffix(".log")
    log = utils.get_logger(log_file=log, log_level=cfg.log_level)
    metas = dict(env=utils.collect_env(), config=cfg.pretty_text)

    # dump environmental information
    log.info(utils.pretty_env(bar="-" * 64))
    log.info("\n{}".format(cfg.pretty_text))

    # build model and dataset
    model = build_detector(cfg.model)
    dataset = build_dataset(cfg.data.train)

    # dump configuration
    os.environ.update(LOCAL_RANK=os.environ.get("LOCAL_RANK", "0"))
    cfg.dump(str(Path(cfg.work_dir).joinpath(Path(cfg_file).name)))

    # start training
    cfg.checkpoint_config.meta = dict(env=utils.collect_env(), CLASSES=dataset.CLASSES)
    train_detector(model, dataset, cfg, devices >= 2, True, timestamp=time, meta=metas)
