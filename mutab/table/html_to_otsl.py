import re
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import List, Tuple

from more_itertools import transpose


@dataclass
class Rank:
    row: int = 0
    col: int = 0


@dataclass
class Span:
    rows: int = 1
    cols: int = 1


@dataclass
class Bbox:
    l: int = 0
    u: int = 0
    r: int = 0
    d: int = 0


@dataclass
class Cell:
    kind: str
    span: Span = field(default_factory=Span)
    rank: Rank = field(default_factory=Rank)
    bbox: Bbox = field(default_factory=Bbox)


PARSERS = dict()


def parser(pattern: str):
    return lambda p: PARSERS.update({pattern: p})


def parse_tag(tag, row, end, seq, box):
    for pattern, p in PARSERS.items():
        if m := re.search(pattern, tag):
            return p(m, row, end, seq, box)


def parse_html(html: List[str], bbox):
    row = []
    seq = []
    end = Cell("R")

    for tag, box in zip(html, bbox):
        end = parse_tag(tag, row, end, seq, box) or end

    return seq


@parser("</tr>")
def tr(tag, row, end, seq, box):
    seq.append([*row, end])
    row.clear()


@parser("<thead>")
def hd(tag, row, end, seq, box):
    return Cell("H")


@parser("<tbody>")
def bd(tag, row, end, seq, box):
    return Cell("B")


@parser("<td")
def td(tag, row, end, seq, box):
    row.append(Cell("D", bbox=Bbox(*box)))


@parser(r"<(eb\d*)>")
def eb(tag, row, end, seq, box):
    row.append(Cell(tag.group(1), bbox=Bbox(*box)))


@parser(r'colspan="(\d+)"')
def cs(tag, row, end, seq, box):
    row[-1].span.cols = int(tag.group(1))


@parser(r'rowspan="(\d+)"')
def rs(tag, row, end, seq, box):
    row[-1].span.rows = int(tag.group(1))


def insert_cell(row: int, col: int, token: str, otsl):
    rank = Rank(row=row, col=col)
    cell = Cell(token, rank=rank)
    otsl[row].insert(col, cell)

    # shift right following cells
    for cell in otsl[row][col + 1 :]:
        cell.rank.col += 1


def expand_cell(row: int, col: int, cell: Cell, otsl):
    rows = range(row, row + cell.span.rows)
    cols = range(col, col + cell.span.cols)
    dlux = (("X", "U"), ("L", cell.kind))
    otsl[row].pop(col)

    # insert DLUX tokens
    for y, x in product(rows, cols):
        if y < len(otsl):
            u, l = int(row == y), int(col == x)
            insert_cell(y, x, dlux[u][l], otsl)

    # copy original bbox
    otsl[row][col].bbox = cell.bbox

    # increment
    return cell.span.cols


def expand_otsl(otsl: List[List[Cell]]):
    for y in reversed(range(len(otsl))):
        x = 0

        # prevent infinite loop
        for cell in list(otsl[y]):
            x += expand_cell(y, x, cell, otsl)

    return otsl


def format_otsl(otsl: List[List[Cell]]):
    h, w = len(otsl), max(map(len, otsl))
    mark = [[False] * w for _ in range(h)]

    # remove dupe cells
    for row, sub in zip(otsl, mark):
        for cell in list(row):
            if sub[cell.rank.col]:
                row.remove(cell)
            else:
                sub[cell.rank.col] = True

    # insert empty cells
    for y, x in product(range(h), range(w)):
        if not mark[y][x]:
            insert_cell(y, x, "eb", otsl)

    return otsl


SOC = ["D"]
EOR = ["R", "H", "B"]


def is_not_empty_row(row: List[Cell]):
    soc = any(v.kind in SOC for v in row)
    eor = all(v.kind in EOR for v in row)
    return soc or eor


def html_to_otsl(html: List[str], bbox: List[Tuple], **kwargs):
    cells = format_otsl(expand_otsl(parse_html(html, bbox)))

    # remove empty rows and columns
    cells = list(filter(is_not_empty_row, transpose(cells)))
    cells = list(filter(is_not_empty_row, transpose(cells)))

    # flatten
    otsl = sum([[v.kind for v in row] for row in cells], [])
    bbox = sum([[v.bbox for v in row] for row in cells], [])

    # return
    bbox = tuple((box.l, box.u, box.r, box.d) for box in bbox)
    return tuple(otsl), bbox, len(cells), max(map(len, cells))


if __name__ == "__main__":
    import json
    from itertools import repeat

    html = json.loads(Path("sample_html.json").read_text())
    otsl = json.loads(Path("sample_otsl.json").read_text())

    # dummy cell bounding boxes
    bbox = repeat((0, 0, 0, 0))

    for html, otsl in zip(html, otsl):
        assert html_to_otsl(html, bbox)[0] == tuple(otsl)
