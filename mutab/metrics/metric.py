import distance
from apted import APTED, Config
from apted.helpers import Tree
from lxml import etree, html


class TableTree(Tree):
    def __init__(self, tag, col=None, row=None, txt=None, *sub):
        super().__init__(tag, *sub)
        self.tag = tag
        self.col = col
        self.row = row
        self.txt = txt


class Custom(Config):
    @staticmethod
    def maximum(*seqs):
        return max(map(len, seqs))

    def normalized_distance(self, *seqs):
        return float(distance.levenshtein(*seqs)) / self.maximum(*seqs)

    def rename(self, node1, node2):
        if node1.tag != node2.tag:
            return 1.0
        if node1.col != node2.col:
            return 1.0
        if node1.row != node2.row:
            return 1.0
        if node1.tag == "td" and (node1.txt or node2.txt):
            return self.normalized_distance(node1.txt, node2.txt)
        return 0.0


class TEDS:
    def __init__(self, ignore_tags=None, struct_only=False):
        self.ignore_tags = ignore_tags
        self.struct_only = struct_only

    def tokenize(self, node, tokens):
        tokens.append("<%s>" % node.tag)
        if node.text is not None:
            tokens += list(node.text)
        for n in node.getchildren():
            self.tokenize(n, tokens)
        if node.tag != "unk":
            tokens.append("</%s>" % node.tag)
        if node.tag != "td" and node.tail is not None:
            tokens += list(node.tail)

    def load_html_tree(self, node, parent=None):
        if node.tag == "td":
            if self.struct_only:
                cell = []
            else:
                tokens = []
                self.tokenize(node, tokens)
                cell = tokens[1:-1].copy()
            col = int(node.attrib.get("colspan", "1"))
            row = int(node.attrib.get("rowspan", "1"))
            sub = TableTree(node.tag, col, row, cell)
        else:
            sub = TableTree(node.tag)
        if parent is not None:
            parent.children.append(sub)
        if node.tag != "td":
            for n in node.getchildren():
                self.load_html_tree(n, sub)
        return sub

    def extract_table(self, text: str, parser):
        tree = html.fromstring(text, parser=parser)
        if not len(tables := tree.xpath("//table")):
            text = "<table>{}</table>".format(text)
            return self.extract_table(text, parser)
        else:
            return next(iter(tables))

    def evaluate(self, pred, real, **kwargs):
        parser = html.HTMLParser(remove_comments=True, encoding="utf-8")
        pred = self.extract_table(pred, parser=parser)
        real = self.extract_table(real, parser=parser)
        if self.ignore_tags:
            etree.strip_tags(pred, *self.ignore_tags)
            etree.strip_tags(real, *self.ignore_tags)
        n_nodes_pred = len(pred.xpath(".//*"))
        n_nodes_real = len(real.xpath(".//*"))
        pred = self.load_html_tree(pred)
        real = self.load_html_tree(real)
        result = APTED(pred, real, Custom()).compute_edit_distance()
        result = 1 - float(result) / max(n_nodes_pred, n_nodes_real)
        return result


if __name__ == "__main__":
    import json

    with open("sample_pred.json") as fp:
        pred_json = json.load(fp)
    with open("sample_real.json") as fp:
        real_json = json.load(fp)
    with open("sample_test.json") as fp:
        test_json = json.load(fp)
    for key in pred_json:
        pred = pred_json[key]
        real = real_json[key]["html"]
        teds = TEDS().evaluate(pred, real)
        print(key, teds)
        assert test_json[key] == teds
