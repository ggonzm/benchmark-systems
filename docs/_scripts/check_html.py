"""Check rendered API declarations and local guide images after a Sphinx build."""

from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

import benchmark_systems.models as models
import benchmark_systems.symbolic as symbolic
import benchmark_systems.animations as animations

HTML = Path(__file__).resolve().parents[1] / "_build" / "html"


class Page(HTMLParser):
    def __init__(self, source):
        super().__init__()
        self.declarations = set()
        self.images = []
        self.text = []
        self.parameter_lists = 0
        self.feed(source)

    def handle_starttag(self, tag, attributes):
        attributes = dict(attributes)
        if tag == "dt" and "id" in attributes:
            self.declarations.add(attributes["id"])
        if tag == "img":
            self.images.append(attributes.get("src", ""))
        if tag == "dl" and "field-list" in attributes.get("class", "").split():
            self.parameter_lists += 1

    def handle_data(self, data):
        self.text.append(data)


def main():
    total = 0
    for filename, module in [("numerical", models), ("symbolic", symbolic),
                             ("animations", animations)]:
        page = Page((HTML / "api" / f"{filename}.html").read_text(encoding="utf-8"))
        expected = {f"{module.__name__}.{name}" for name in module.__all__}
        missing = expected - page.declarations
        assert not missing, f"API declarations missing: {sorted(missing)}"
        text = "".join(page.text)
        for marker in (".. py:function::", ".. py:class::", ".. rubric::", ":param ", ":type "):
            assert marker not in text, f"Unparsed reStructuredText in {filename}: {marker}"
        assert page.parameter_lists > 0, f"No formatted parameters in {filename}"
        total += len(expected)
    image_count = 0
    for filename in ("models", "numerical", "symbolic", "quickstart", "animations",
                     "api/numerical", "api/symbolic", "api/animations"):
        path = HTML / f"{filename}.html"
        page = Page(path.read_text(encoding="utf-8"))
        for source in page.images:
            url = urlsplit(source)
            if not url.scheme and not url.netloc:
                assert (path.parent / unquote(url.path)).is_file(), f"Missing image: {source}"
                image_count += 1
    print(f"Verified {total} rendered public API declarations and {image_count} local guide images.")


if __name__ == "__main__":
    main()
