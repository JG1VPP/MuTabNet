import re
from typing import Dict, List


class Revisor:
    def __init__(
        self,
        SOC: List[str],
        EOC: List[str],
        template: str,
        patterns: Dict[str, Dict[str, str]],
    ):
        assert isinstance(SOC, list)
        assert isinstance(EOC, list)

        self.SOC = SOC
        self.EOC = EOC

        assert isinstance(template, str)
        assert isinstance(patterns, dict)

        self.template = template
        self.patterns = patterns

    def merge(self, html, cell):
        contents = iter(cell)
        internal = False
        restored = []
        for idx, el in enumerate(html):
            if el in self.SOC:
                internal = True
            if internal and el in self.EOC:
                ch = "".join(next(contents, ""))
                el = el.replace("</", f"{ch}</")
                internal = False
            restored.append(el)
        return "".join(restored)

    def clean(self, text):
        for pattern, subpatterns in self.patterns.items():
            section = re.search(pattern, text)
            if section is None:
                continue
            original = section = section.group()
            for pattern, replace in subpatterns.items():
                section = re.sub(pattern, replace, section)
            text = text.replace(original, section)
        return self.template.format(text)

    def __call__(self, html, cell, **kwargs):
        return self.clean(self.merge(html, cell))
