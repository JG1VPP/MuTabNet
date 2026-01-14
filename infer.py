import argparse
from multiprocessing import Pool
from pathlib import Path
from pickle import dumps

from mmengine import Config
from mmengine.device import get_device
from mmengine.runner import Runner, load_checkpoint
from more_itertools import flatten
from tqdm import tqdm

from mutab.score import TEDS
from mutab.utils import build


def options():
    args = argparse.ArgumentParser()

    args.add_argument("--config", type=str, required=True)
    args.add_argument("--weight", type=str, required=True)
    args.add_argument("--split", type=str, required=True)
    args.add_argument("--store", type=str, required=True)

    return args.parse_args()


def conduct(model, loader: dict, split: str):
    loader.dataset.filter_cfg.update(split=split)

    pool = Pool()
    model = model.eval()
    metric = TEDS(prefix="full")

    loader = tqdm(Runner.build_dataloader(loader))
    result = flatten(map(model.test_step, loader))

    result = tuple(result)
    result = tuple(pool.map(metric._teds, result))

    scores = metric.compute_metrics(result)
    return dict(scores=scores, data=result)


def process(config: str, weight: str, split: str, store: str):
    # config
    config = Config.fromfile(config)

    # model
    model = build(config.model).to(get_device())
    load_checkpoint(model, weight, strict=False)

    # infer
    result = conduct(model, config.test_dataloader, split)

    # store
    path = Path(store).expanduser()
    path.write_bytes(dumps(result))


if __name__ == "__main__":
    process(**vars(options()))
