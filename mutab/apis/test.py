import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import List, Optional

from mmdet.apis import init_detector
from tqdm import tqdm

from mutab.metrics import TEDS


def score(item, truth, ignore: Optional[List[str]]):
    teds_full = TEDS(ignore, struct_only=False)
    teds_html = TEDS(ignore, struct_only=True)
    file_name = os.path.basename(item["path"])
    if file_name not in truth:
        return None
    item.update(real=truth[file_name]["html"])
    item.update(type=truth[file_name]["type"])
    scores = {}
    scores.update(full=teds_full.evaluate(**item))
    scores.update(html=teds_html.evaluate(**item))
    item.update(TEDS=scores)
    return (file_name, item)


def worker(n: int, paths: List[str], cfg: str, ckpt: str, truth):
    model = init_detector(config=cfg, checkpoint=ckpt, device=n)
    items = map(model.predict, tqdm(list(paths), disable=n > 0))
    final = partial(score, truth=truth, ignore=model.cfg.ignore)
    with ProcessPoolExecutor() as pool:
        return list(pool.map(final, items))


def evaluate(paths: List[List[str]], cfg: str, ckpt: str, truth):
    with ProcessPoolExecutor(len(paths)) as pool:
        process = partial(worker, cfg=cfg, ckpt=ckpt, truth=truth)
        results = list(pool.map(process, *zip(*enumerate(paths))))
        return dict(filter(None, sum(results, [])))
