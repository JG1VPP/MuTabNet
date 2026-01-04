import argparse
import json
import pickle
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dacite import from_dict
from more_itertools import take
from ruamel.yaml import YAML
from tqdm import tqdm

TAB = "\t"


@dataclass
class Part:
    image: str
    jsonl: str


@dataclass
class Load:
    parts: List[Part]


@dataclass
class Dump:
    pkl: str


@dataclass
class Range:
    min: int = 0
    max: int = sys.maxsize


@dataclass
class SeqLen:
    html: Range = field(default_factory=Range)
    cell: Range = field(default_factory=Range)


@dataclass
class Preprocess:
    load: Load
    dump: Dump
    replace: Dict[Tuple, str]
    seq_len: Optional[SeqLen] = None


def options():
    args = argparse.ArgumentParser()
    args.add_argument("cfg")
    args = args.parse_args()

    data = YAML().load(path(args.cfg))
    return from_dict(Preprocess, data)


def path(root, *levels):
    path = Path(root).joinpath(*levels)
    return path.expanduser().absolute()


def index_thead(tokens):
    if "<thead>" in tokens:
        thead = take(tokens.index("</thead>"), tokens)
        return list(range(list(thead).count("</td>")))

    else:
        return range(0)


def merge_close(tokens):
    source = iter(tokens)
    target = list()

    for token in source:
        close = take(token == "<td>", source)
        target.append(token + "".join(close))

    return target


def format_html(params, tokens, cell):
    source = iter(cell)
    target = list()

    for token in tokens:
        if token.startswith("<td"):
            cell = next(source)

            if not cell.get("bbox"):
                token = tuple(cell.get("tokens"))
                token = params.replace.get(token)

        target.append(token)

    return target


def count_cell_tokens(params, tokens):
    targets = ["<td", "<td></td>", *params.replace.values()]
    return sum(map(lambda tag: int(tag in targets), tokens))


def unbold(content, ok: bool):
    def remove(tag):
        return ok or tag not in ("<b>", "</b>")

    return TAB.join(filter(remove, content)).strip().split(TAB)


def dump_cell(cell, head: bool, dump: list):
    if bbox := cell.get("bbox"):
        text = unbold(cell["tokens"], not head)
        dump.append(dict(bbox=bbox, text=text))


def dump_item(params, part, split, name, html, cell):
    head = index_thead(html)
    name = path(part.image, split, name)
    html = format_html(params, merge_close(html), cell)
    assert len(cell) == count_cell_tokens(params, html)

    dump = []

    for num, cell in enumerate(cell):
        dump_cell(cell, num in head, dump=dump)

    return dict(img_path=str(name), html=html, cell=dump)


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


def fix_json(text):
    return text.replace(", ]", "]")


def parse(params, part, row, splits):
    name = row["filename"]
    html = row["html"]["structure"]["tokens"]
    cell = row["html"]["cells"]
    page = row["split"]

    item = dict(name=name, html=html, cell=cell)
    item = dump_item(params, part, page, **item)

    if is_long_sample(params, html, cell):
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
