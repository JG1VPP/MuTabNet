import re
from abc import ABC, abstractmethod
from typing import Dict, List

from mutab.models.factory import REVISORS, build_revisor
from mutab.syntax import otsl_to_html


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
class TableCombine(RevisorModule):
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

        # during test
        if html is None:
            return results

        # loop states
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
class TableReplace(RevisorModule):
    def __init__(self, replace: Dict[str, str]):
        assert isinstance(replace, dict)

        # patterns
        self.replace = replace.items()

    def process(self, results):
        html = results.get("html")

        # during test
        if html is None:
            return results

        # search text
        for pattern in self.replace:
            html = re.sub(*pattern, html)

        # update html
        results.update(html=html)

        return results


@REVISORS.register_module()
class ToHTML(RevisorModule):
    def process(self, results):
        html = results.get("html")

        # in service
        if html is None:
            return results

        # update html
        html = otsl_to_html(html)
        results.update(html=html)

        return results
