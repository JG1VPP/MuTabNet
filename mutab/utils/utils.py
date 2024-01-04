import os

import cv2
import mmcv.utils as utils


def get_logger(**kwargs):
    return utils.get_logger("mmdet", **kwargs)


def collect_env():
    return dict(**utils.collect_env(), commit=utils.get_git_hash())


def pretty_env(bar: str):
    contents = list(f"{k}: {v}" for k, v in collect_env().items())
    return "\n".join(["", bar] + contents + [bar, ""])


def visualize_bbox(bbox, path, save, **kwargs):
    img = cv2.imread(path)
    for x, y, w, h in bbox:
        a = int(x - w / 2), int(y - h / 2)
        b = int(x + w / 2), int(y + h / 2)
        img = cv2.rectangle(img, a, b, (0, 0, 255), thickness=1)
    cv2.imwrite(os.path.join(save, os.path.basename(path)), img)
