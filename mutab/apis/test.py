from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import List, Optional

from mmcv import Config
from mmdet.apis import init_detector
from more_itertools import flatten, transpose
from tqdm import tqdm

from mutab.metrics import TEDS


def test(item, truth, ignore: Optional[List[str]]):
    teds_full = TEDS(ignore, struct_only=False)
    teds_html = TEDS(ignore, struct_only=True)
    file_name = Path(item["path"]).name
    if file_name not in truth:
        return None
    item.update(real=truth[file_name]["html"])
    item.update(type=truth[file_name]["type"])
    scores = {}
    scores.update(full=teds_full.evaluate(**item))
    scores.update(html=teds_html.evaluate(**item))
    item.update(TEDS=scores)
    return (file_name, item)


def worker(n: int, paths: List[str], conf: Config, ckpt: str, truth):
    model = init_detector(conf, checkpoint=ckpt, device=n)
    items = map(model.predict, tqdm(list(paths), disable=n))
    score = partial(test, truth=truth, ignore=conf.ignore)
    with ProcessPoolExecutor() as runtime:
        return list(runtime.map(score, items))


def evaluate(paths: List[List[str]], conf: Config, ckpt: str, truth):
    with ProcessPoolExecutor(len(paths)) as runtime:
        process = partial(worker, conf=conf, ckpt=ckpt, truth=truth)
        results = runtime.map(process, *transpose(enumerate(paths)))
        return dict(filter(None, flatten(results)))


def rescore(results, conf: Config, truth):
    with ProcessPoolExecutor() as runtime:
        score = partial(test, truth=dict(truth), ignore=conf.ignore)
        items = tqdm(runtime.map(score, results), total=len(results))
        return dict(filter(None, items))
