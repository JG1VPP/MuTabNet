import argparse
from pathlib import Path

import numpy as np
from more_itertools import take

from mutab.apis import init_detector
from mutab.utils import heatmap


def options():
    args = argparse.ArgumentParser()
    args.add_argument("--ckpt", type=str, default="latest.pth")
    args.add_argument("--save", type=str, default="{:02d}.png")
    args.add_argument("--conf", type=str, required=True)
    args.add_argument("--shot", type=int, default=10)
    args.add_argument("--crop", type=int, default=80)
    args.add_argument("--tone", type=int, default=256)
    args.add_argument("--opaq", type=int, default=160)
    args.add_argument("--cmap", type=str, default="hot")
    args.add_argument("--bbox", type=str, default="blue")
    args.add_argument("--line", type=int, default=2)
    args.add_argument("--pool", type=int, default=8)
    args.add_argument("source", type=str)
    return args.parse_args()


def process(args, items=[]):
    root = Path(args.ckpt).parent.expanduser()
    scan = init_detector(args.conf, args.ckpt)
    attn = []

    def hook(p):
        attn.clear()
        attn.extend(p.cpu().mean(dim=[0, 1]))
        return p

    scan.decoder.cell.dec[-1].att2.hook = hook
    item = scan.predict(args.source)
    bbox = item.get("bbox", [])
    cell = item.get("cell", [])
    meta = item.get("meta")
    attn = iter(attn)

    for n, (xywh, cell) in enumerate(take(args.shot, zip(bbox, cell))):
        size = tuple(num // args.pool for num in meta["pad_shape"][:2])
        snap = np.mean(take(len(cell) + 1, attn), axis=0).reshape(size)
        shot = heatmap(args.source, *xywh, attn=snap, **vars(args))
        shot.save(root.joinpath(args.save.format(n)), format="PNG")


if __name__ == "__main__":
    process(options())
