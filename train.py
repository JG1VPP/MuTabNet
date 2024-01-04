#!/usr/bin/python3
import argparse

from mmcv import Config
from mmcv.runner import init_dist
from torch.multiprocessing import set_start_method

from mutab.apis import train


def main():
    args = argparse.ArgumentParser()
    args.add_argument("config")
    args.add_argument("--work-dir", required=True)
    args.add_argument("--launcher", required=False)
    args, _ = args.parse_known_args()

    cfg = Config.fromfile(args.config)
    cfg.update(**vars(args))
    set_start_method("fork")

    if args.launcher is not None:
        init_dist(args.launcher, **cfg.dist_params)

    train(cfg, args.config)


if __name__ == "__main__":
    main()
