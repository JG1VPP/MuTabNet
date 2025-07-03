import re
from abc import ABC, abstractmethod
from functools import partial
from typing import Dict, List

from lxml.etree import tostring
from lxml.html import HTMLParser, fromstring

from mutab.models.factory import REVISORS, build_revisor


class RevisorModule(ABC):
    @abstractmethod
    def process(self, results):
        pass

    def __call__(self, html=None, cell=None, **kwargs):
        return self.process(dict(html=html, cell=cell)).get("html")


@REVISORS.register_module()
class TableRevisor(RevisorModule):
    def __init__(self, pipeline: List[Dict]):
        self.sub = tuple(map(build_revisor, pipeline))

    def process(self, results):
        for sub in self.sub:
            results = sub.process(results)

        return results


@REVISORS.register_module()
class Combine(RevisorModule):
    def __init__(self, SOC: List[str], EOC: List[str]):
        assert isinstance(SOC, list)
        assert isinstance(EOC, list)

        assert all(isinstance(v, str) for v in SOC)
        assert all(isinstance(v, str) for v in EOC)

        # tokens
        self.SOC = SOC
        self.EOC = EOC

    def process(self, results):
        html = results.get("html")
        cell = results.get("cell")

        # in service
        if html is None:
            return results

        # states
        contents = iter(cell)
        internal = False
        combined = []

        for el in html:
            # <td
            if el in self.SOC:
                internal = True

            # combine tag and cell content
            if internal and el in self.EOC:
                ch = "".join(next(contents, ""))
                el = el.replace("</", f"{ch}</")
                internal = False

            combined.append(el)

        # update html
        results.update(html="".join(combined))

        return results


@REVISORS.register_module()
class Replace(RevisorModule):
    def __init__(self, path: str, maps: Dict, encode: str):
        assert isinstance(path, str)
        assert isinstance(maps, dict)
        assert isinstance(encode, str)

        # patterns
        self.path = path
        self.maps = maps.items()

        # parser
        parser = HTMLParser(encoding=encode)

        # text/html
        self.load = partial(fromstring, parser=parser)
        self.dump = partial(tostring, encoding=encode)

    def replace(self, html: str):
        for entity in self.maps:
            html = re.sub(*entity, html)

        return html

    def process(self, results):
        html = results.get("html")

        # in service
        if html is None:
            return results

        # parse html
        root = self.load(html)

        # search node
        for node in root.xpath(self.path):
            dump = self.dump(node).decode()
            post = self.replace(dump)
            html = html.replace(dump, post)

        # update html
        results.update(html=html)

        return results
