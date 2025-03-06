import mmcv.utils as utils
import numpy as np
from cmap import Colormap
from PIL import Image, ImageDraw


def get_logger(**kwargs):
    return utils.get_logger("mmdet", **kwargs)


def collect_env():
    return dict(**utils.collect_env(), commit=utils.get_git_hash())


def pretty_env(bar: str):
    contents = list(f"{k}: {v}" for k, v in collect_env().items())
    return "\n".join(["", bar] + contents + [bar, ""])


def heatmap(
    path: str,
    x: int,
    y: int,
    w: int,
    h: int,
    attn,
    crop: int,
    tone: int,
    opaq: int,
    cmap: str,
    bbox: str,
    line: int,
    **kwargs,
):
    view = Image.open(path)
    heat = Colormap(cmap)(tone * attn, N=tone)
    heat = Image.fromarray(np.multiply(tone, heat).astype(np.uint8))
    zoom = np.divide(np.array(view.size), np.array(heat.size)).max()
    heat = heat.resize(map(round, zoom * np.array(heat.size)))
    heat = heat.crop((0, 0, heat.width, crop))
    view = view.crop((0, 0, view.width, crop))
    heat.putalpha(opaq)
    view.paste(heat, mask=heat)
    draw = ImageDraw.Draw(view)
    lt = round(x - w / 2), round(y - h / 2)
    rb = round(x + w / 2), round(y + h / 2)
    draw.rectangle((lt, rb), outline=bbox, width=line)
    return view.convert("RGB")
