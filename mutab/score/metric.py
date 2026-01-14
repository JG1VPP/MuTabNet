from operator import itemgetter
from statistics import mean

import distance
from apted import APTED, Config
from apted.helpers import Tree
from lxml import etree, html
from mmengine.evaluator import BaseMetric
from mmengine.registry import METRICS
from tqdm import tqdm


class TableTree(Tree):
    def __init__(self, tag, col, row, txt, *sub):
        super().__init__(tag, *sub)
        self.tag = tag
        self.col = col
        self.row = row
        self.txt = txt

    @property
    def struct(self):
        return self.tag, self.col, self.row


class Custom(Config):
    def rename_text(self, txt1, txt2):
        dist = distance.levenshtein(txt1, txt2)
        return dist / max(len(txt1), len(txt2))

    def rename(self, node1, node2):
        if node1.struct != node2.struct:
            return 1.0

        if node1.tag == "td" and (node1.txt or node2.txt):
            return self.rename_text(node1.txt, node2.txt)

        return 0.0


@METRICS.register_module()
class TEDS(BaseMetric):
    OUTPUTS = "outputs"
    TARGETS = "targets"

    def __init__(self, ignore=None, **kwargs):
        super().__init__(**kwargs)

        self.ignore = ignore or []

    def process(self, data_batch, data_samples):
        self.results.extend(map(self._teds, data_samples))

    def compute_metrics(self, results: list):
        html = mean(map(itemgetter("html"), results))
        full = mean(map(itemgetter("full"), results))

        return dict(html=html, full=full)

    def _teds(self, result):
        y = result[self.OUTPUTS][self.prefix]
        t = result[self.TARGETS][self.prefix]

        html = self.score(y, t, struct=True)
        full = self.score(y, t, struct=False)

        return dict(data=result, html=html, full=full)

    def score(self, pred, real, struct=False, **kwargs):
        parser = html.HTMLParser(encoding="utf-8")

        pred = self.extract_table(pred, parser=parser)
        real = self.extract_table(real, parser=parser)

        etree.strip_tags(pred, *self.ignore)
        etree.strip_tags(real, *self.ignore)

        num_tags_pred = len(pred.xpath(".//*"))
        num_tags_real = len(real.xpath(".//*"))

        pred = self.html_to_tree(pred, struct=struct)
        real = self.html_to_tree(real, struct=struct)

        result = APTED(pred, real, Custom()).compute_edit_distance()
        return 1 - float(result) / max(num_tags_pred, num_tags_real)

    def extract_table(self, text: str, parser):
        tree = html.fromstring(text, parser=parser)

        if not len(tables := tree.xpath("//table")):
            text = "<table>{}</table>".format(text)
            return self.extract_table(text, parser)
        else:
            return next(iter(tables))

    def html_to_tree(self, node, struct: bool, parent=None):
        col = int(node.attrib.get("colspan", 1))
        row = int(node.attrib.get("rowspan", 1))

        if node.tag == "td" and not struct:
            sub = self.tokenize(node)[1:-1]
        else:
            sub = []

        sub = TableTree(node.tag, col, row, sub)

        if parent is not None:
            parent.children.append(sub)

        if node.tag != "td":
            for n in node.getchildren():
                self.html_to_tree(n, struct, sub)

        return sub

    def tokenize(self, node) -> list[str]:
        tokens = [f"<{node.tag}>"]

        if node.text is not None:
            tokens.extend(node.text)

        for n in node.getchildren():
            tokens.extend(self.tokenize(n))

        tokens.append(f"</{node.tag}>")

        if node.tail is not None:
            tokens += node.tail

        return tokens


if __name__ == "__main__":
    import json
    from pathlib import Path

    pred_json = json.loads(Path("sample_pred.json").read_text())
    real_json = json.loads(Path("sample_real.json").read_text())
    test_json = json.loads(Path("sample_test.json").read_text())

    for key in tqdm(pred_json):
        pred = pred_json[key]
        real = real_json[key]["html"]
        test = test_json[key]

        assert test == TEDS(prefix="html").score(pred, real)
