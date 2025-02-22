import argparse
import json
import lzma
import os
import pickle
import time
from datetime import timedelta as td
from glob import glob
from pathlib import Path

import numpy as np
from more_itertools import divide
from torch.multiprocessing import set_start_method

from mutab.apis import evaluate

EASY = "simple"
HARD = "complex"


def main():
    args = argparse.ArgumentParser()
    args.add_argument("--gpus", type=int, default=4)
    args.add_argument("--ckpt", type=str, default="latest.pth")
    args.add_argument("--save", type=str, default="results.xz")
    args.add_argument("--json", type=str, required=True)
    args.add_argument("--conf", type=str, required=True)
    args.add_argument("--path", type=str, required=True)
    args = args.parse_args()

    root = Path(args.ckpt).parent.expanduser()

    with open(args.json) as f:
        jsonl_ground_truth = json.load(f)

    set_start_method("spawn")
    count = time.perf_counter()
    paths = divide(args.gpus, glob(os.path.join(args.path, "*.png")))
    items = evaluate(paths, args.conf, args.ckpt, jsonl_ground_truth)
    count = td(seconds=time.perf_counter() - count) / td(hours=1)

    easy = list(v for v in items.values() if v["type"] == EASY)
    hard = list(v for v in items.values() if v["type"] == HARD)

    summary = {}
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
        pickle.dump(dict(results=items, summary=summary, **vars(args)), f)


if __name__ == "__main__":
    main()
