import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import jsonlines
import toolz.dicttoolz as dicttool
from dacite import from_dict
from more_itertools import take
from ruamel.yaml import YAML
from tqdm import tqdm

FTN = "FinTabNet"
PTN = "PubTabNet"

EASY = "simple"
HARD = "complex"


@dataclass
class Load:
    dir: str
    jsonl: List[str]


@dataclass
class Dump:
    dir: str
    json: str
    split: str


@dataclass
class Range:
    min: int = 0
    max: int = sys.maxsize


@dataclass
class SeqLen:
    html: Range = field(default_factory=Range)
    cell: Range = field(default_factory=Range)


@dataclass
class TabNet:
    load: Load
    dump: Dump
    type: Literal[FTN, PTN]
    replace: Dict[Tuple, str]
    samples: Optional[int] = None
    seq_len: Optional[SeqLen] = None


def path(root, *levels):
    path = Path(root).joinpath(*levels)
    return path.expanduser().absolute()


def merge_tokens(tokens):
    source = iter(tokens)
    target = list()

    for token in source:
        close = take(token == "<td>", source)
        target.append(token + "".join(close))

    return target


def insert_empty(params, tokens, cell):
    source = iter(cell)
    target = list()

    for token in tokens:
        if token.startswith("<td"):
            cell = next(source)

            if not cell.get("bbox"):
                token = tuple(cell["tokens"])
                token = params.replace[token]

        target.append(token)

    return target


def get_thead_item_idx(tokens):
    if "<thead>" in tokens:
        thead = take(tokens.index("</thead>"), tokens)
        return list(range(list(thead).count("</td>")))

    else:
        return range(0)


def count_merged_tokens(params, tokens):
    targets = ["<td", "<td></td>", *params.replace.values()]
    return sum(map(lambda tag: int(tag in targets), tokens))


def unbold(content, ok: bool):
    def remove(tag):
        return ok or tag not in ("<b>", "</b>")

    return "\t".join(filter(remove, content))


def print_cell(cell, header: bool, f):
    if bbox := cell.get("bbox"):
        bbox = ",".join(map(str, map(int, bbox)))
        text = unbold(cell["tokens"], not header)

    else:
        bbox = "0,0,0,0"
        text = "<UKN>"

    print(f"{bbox}<;>{text}", file=f)


def print_item(params, split, item, html, cell, f):
    head = get_thead_item_idx(html)
    name = path(params.load.dir, split, item["filename"])
    html = insert_empty(params, merge_tokens(html), cell)
    assert len(cell) == count_merged_tokens(params, html)

    print(name.resolve(), file=f)
    print(",".join(html), file=f)

    for n, cell in enumerate(cell):
        print_cell(cell, n in head, f)


def combine_cell(params, html, cell):
    if params.type == PTN or cell.get("bbox"):
        cell = "".join(cell.get("tokens", ""))
        return html.replace("</", f"{cell}</")

    else:
        return html


def combine_html(params, html, cell):
    source = iter(cell)
    target = []

    for html in html:
        if html.endswith("</td>"):
            html = combine_cell(params, html, next(source))

        target.append(html)

    return "".join(target)


def is_long_sample(params, html, cell):
    ok = True

    if params.seq_len is not None:
        # length limit of html tokens
        hmin = params.seq_len.html.min
        hmax = params.seq_len.html.max

        # length limit of cell tokens
        cmin = params.seq_len.cell.min
        cmax = params.seq_len.cell.max

        ok = ok and (hmin <= len(html) <= hmax)
        ok = ok and (cmin <= len(cell) <= cmax)

    return ok


def open_annon_file(params, split, item):
    name = Path(item["filename"]).with_suffix(suffix=".txt")
    return path(params.dump.dir, split, name).open(mode="w")


def parse(params, split, item):
    html = item["html"]["structure"]["tokens"]
    cell = item["html"]["cells"]
    name = item["filename"]

    if is_long_sample(params, html, cell):
        with open_annon_file(params, split, item) as f:
            print_item(params, split, item, html, cell, f)

        if split == params.dump.split:
            lev = (EASY, HARD)[">" in html]
            txt = combine_html(params, html, cell)
            return {name: dict(html=txt, type=lev)}

    return {}


def splits(params):
    splits = defaultdict(list)

    for name in params.load.jsonl:
        with jsonlines.open(path(name)) as f:
            for item in tqdm(f, desc=name):
                splits[item["split"]].append(item)

    drop = partial(take, params.samples)
    return dicttool.valmap(drop, splits)


def process(params):
    test = {}

    for split, items in splits(params).items():
        name = path(params.dump.dir, split)
        name.mkdir(parents=True, exist_ok=True)

        for item in tqdm(items, desc=split):
            test.update(parse(params, split, item))

    with path(params.dump.json).open(mode="w") as f:
        json.dump(test, f)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("config")
    args = args.parse_args()

    with path(args.config).open() as f:
        params = from_dict(TabNet, YAML().load(f))

    process(params)
