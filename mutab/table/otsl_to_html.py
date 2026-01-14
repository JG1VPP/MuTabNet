import re
from itertools import takewhile, zip_longest
from pathlib import Path
from typing import List

from more_itertools import ilen, split_after, transpose

HEAD = "<thead>"
BODY = "<tbody>"

DAEH = "</thead>"
YDOB = "</tbody>"


def is_end_of_row(cell: str):
    return cell in ["R", "H", "B"]


def is_expand_row(cell: str):
    return cell in "U"


def is_expand_col(cell: str):
    return cell == "L"


def is_head_break(cell: str):
    return cell in "H"


def is_body_break(cell: str):
    return cell in "B"


def format_otsl(otsl: List[str]):
    # fill lacking cells
    otsl = tuple(split_after(otsl, is_end_of_row))
    cols = tuple(zip_longest(*otsl, fillvalue=""))
    rows = tuple(transpose(cols))

    return rows, cols


PARSERS = dict()


def parser(pattern: str):
    return lambda p: PARSERS.update({pattern: p})


def span(tag, y: int, x: int, rows, cols, html, full):
    rs = ilen(takewhile(is_expand_row, cols[x][y + 1 :]))
    cs = ilen(takewhile(is_expand_col, rows[y][x + 1 :]))
    td = []

    # create <TD>
    td.append(f"<{tag}")
    td.append(f' colspan="{cs + 1}"' if cs else None)
    td.append(f' rowspan="{rs + 1}"' if rs else None)
    td.append(">")
    td.append("</td>")
    td = list(filter(None, td))

    if rs or cs:
        html.extend(td)
    else:
        html.append("".join(td))


@parser("D")
def td(cell, y: int, x: int, rows, cols, html, full):
    span("td", y, x, rows, cols, html, full)


@parser(r"eb\d*")
def eb(cell, y: int, x: int, rows, cols, html, full):
    span(cell.group(), y, x, rows, cols, html, full)


@parser("R")
def tr(cell, y: int, x: int, rows, cols, html, full):
    html.append("</tr>")


@parser("H")
def hd(cell, y: int, x: int, rows, cols, html, full):
    hd = ilen(takewhile(is_head_break, cols[x][y + 1 :]))
    html.append("</tr>")

    if not hd:
        full.append(HEAD)
        full.extend(html)
        full.append(DAEH)
        html.clear()


@parser("B")
def bd(cell, y: int, x: int, rows, cols, html, full):
    bd = ilen(takewhile(is_body_break, cols[x][y + 1 :]))
    html.append("</tr>")

    if not bd:
        full.append(BODY)
        full.extend(html)
        full.append(YDOB)
        html.clear()


def parse_cell(y: int, x: int, rows, cols, html, full):
    for pattern, p in PARSERS.items():
        if m := re.search(pattern, rows[y][x]):
            p(m, y, x, rows, cols, html, full)
            break


def otsl_to_html(otsl: List[str]):
    rows, cols = format_otsl(otsl)
    html = []
    full = []

    h = len(rows)
    w = len(cols)

    for y in range(h):
        html.append("<tr>")

        for x in range(w):
            parse_cell(y, x, rows, cols, html, full)

    return tuple([*full, *html])


if __name__ == "__main__":
    import json

    html = json.loads(Path("sample_html.json").read_text())
    otsl = json.loads(Path("sample_otsl.json").read_text())

    for html, otsl in zip(html, otsl):
        assert otsl_to_html(otsl) == tuple(html)
