import argparse
import json
import pickle
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from dacite import from_dict
from more_itertools import ilen, iter_index, take
from ruamel.yaml import YAML
from tqdm import tqdm

TAB = "\t"

BOLD = "<b>"
DLOB = "</b>"


@dataclass
class Part:
    image: str
    jsonl: str


@dataclass
class Load:
    parts: list[Part]


@dataclass
class Dump:
    pkl: str


@dataclass
class Range:
    min: int = 0
    max: int = sys.maxsize


@dataclass
class Length:
    html: Range = field(default_factory=Range)
    cell: Range = field(default_factory=Range)


@dataclass
class Preprocess:
    load: Load
    dump: Dump
    replace: dict[tuple, str]


def options():
    args = argparse.ArgumentParser()
    args.add_argument("cfg")
    args = args.parse_args()

    data = YAML().load(path(args.cfg))
    return from_dict(Preprocess, data)


def path(root, *levels):
    path = Path(root).joinpath(*levels)
    return path.expanduser().absolute()


def count_head(html):
    size = next(iter_index(html, "</thead>"), 0)
    head = iter_index(take(size, html), "</td>")

    return ilen(head)


def tokenize(params, tokens):
    html = iter(tokens)
    half = list()

    for tag in html:
        close = take(tag == "<td>", html)
        half.append(tag + "".join(close))

    return half


def classify(params, tokens, grid):
    grid = iter(grid)
    html = list()

    for tag in tokens:
        if tag.startswith("<td"):
            if not (cell := next(grid)).get("bbox"):
                value = tuple(cell.get("tokens"))
                value = params.replace.get(value)
                tag = tag.replace("td", value, 1)

        html.append(tag)

    return html


def unbold(tokens, is_head_cell: bool):
    if is_head_cell:
        tokens = filter(BOLD.__ne__, tokens)
        tokens = filter(DLOB.__ne__, tokens)

    return TAB.join(tokens).strip().split(TAB)


def dump_item(params, part, split, name, html, grid):
    name = str(path(part.image, split, name))
    head = count_head(html)
    dump = []

    for n, cell in enumerate(grid):
        if bbox := cell.get("bbox"):
            text = unbold(cell["tokens"], n < head)
            dump.append(dict(bbox=bbox, text=text))

    html = classify(params, tokenize(params, html), grid)
    assert len(dump) == sum("<td" in tag for tag in html)

    return dict(img_path=name, html=html, cell=dump)


def fix_json(text):
    return text.replace(", ]", "]")


def parse(params, part, row, splits):
    name = row["filename"]
    html = row["html"]["structure"]["tokens"]
    grid = row["html"]["cells"]
    page = row["split"]

    item = dict(name=name, html=html, grid=grid)
    item = dump_item(params, part, page, **item)

    splits[page].append(item)


def jsonl(params, part, splits):
    with path(part.jsonl).open() as f:
        for line in tqdm(f, desc=part.jsonl):
            line = json.loads(fix_json(line))
            parse(params, part, line, splits)


def process(params):
    table_splits = defaultdict(list)

    for part in params.load.parts:
        jsonl(params, part, table_splits)

    data = pickle.dumps(dict(table_splits))
    path(params.dump.pkl).write_bytes(data)


if __name__ == "__main__":
    process(options())
