import argparse
import json
import lzma
import pickle
import time
from datetime import timedelta as td
from pathlib import Path

import numpy as np
from mmcv import Config
from more_itertools import divide
from torch.multiprocessing import set_start_method

from mutab.apis import evaluate, rescore
from mutab.utils import collect_env

EASY = "simple"
HARD = "complex"


def options():
    args = argparse.ArgumentParser()
    args.add_argument("--gpus", type=int, default=4)
    args.add_argument("--rate", type=int, default=None)
    args.add_argument("--ckpt", type=str, default="latest.pth")
    args.add_argument("--save", type=str, default="results.xz")
    args.add_argument("--load", type=str, required=False)
    args.add_argument("--json", type=str, required=True)
    args.add_argument("--conf", type=str, required=True)
    args.add_argument("--path", type=str, required=True)
    args.add_argument("--fork", type=str, default="spawn")
    return args.parse_args()


def process(args, items=[]):
    conf = Config.fromfile(args.conf)
    cond = dict(**collect_env(), **vars(args))
    root = Path(args.ckpt).parent.expanduser()

    with open(args.json) as f:
        truth = json.load(f)

    if args.load is not None:
        with lzma.open(args.load) as f:
            items = pickle.load(f)["results"].values()

    def infer(conf: Config, args, truth):
        conf.model.decoder.update(max_spd=args.rate)
        paths = divide(args.gpus, Path(args.path).rglob("*.png"))
        return evaluate(paths, conf, ckpt=args.ckpt, truth=truth)

    set_start_method(args.fork)
    count = time.perf_counter()
    items = rescore(items, conf, truth) or infer(conf, args, truth)
    count = td(seconds=time.perf_counter() - count) / td(hours=1.0)

    easy = list(v for v in items.values() if v["type"] == EASY)
    hard = list(v for v in items.values() if v["type"] == HARD)

    def time_stamp(item, key):
        return times[key] if (times := item.get("time")) is not None else 0

    def count_time(item, key):
        return 1e-9 * (time_stamp(item, key) - time_stamp(item, "init"))

    time_html = np.mean([count_time(v, "html") for v in items.values()])
    time_bbox = np.mean([count_time(v, "bbox") for v in items.values()])
    time_cell = np.mean([count_time(v, "cell") for v in items.values()])

    summary = {}
    summary.update(time=dict(html=time_html, bbox=time_bbox, cell=time_cell))
    summary.update(html=np.mean([v["TEDS"]["html"] for v in items.values()]))
    summary.update(full=np.mean([v["TEDS"]["full"] for v in items.values()]))
    summary.update(easy=np.mean([v["TEDS"]["full"] for v in easy]))
    summary.update(hard=np.mean([v["TEDS"]["full"] for v in hard]))

    with open(root.joinpath("{}.log".format(args.save)), "w") as f:
        print(f"{len(items)} samples in {count:.2f} hours:", file=f)
        print(f"AVG TEDS html score: {summary['html']:.4f}", file=f)
        print(f"AVG TEDS full score: {summary['full']:.4f}", file=f)
        print(f"AVG TEDS easy score: {summary['easy']:.4f}", file=f)
        print(f"AVG TEDS hard score: {summary['hard']:.4f}", file=f)

    with lzma.open(root.joinpath(args.save), "wb") as f:
        pickle.dump(dict(results=items, summary=summary, **cond), f)


if __name__ == "__main__":
    process(options())
