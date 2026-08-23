from html.parser import HTMLParser
from pathlib import Path
import xml.etree.ElementTree as ElementTree


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SITE_ROOT = REPOSITORY_ROOT / "docs"


class SiteParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: set[str] = set()
        self.local_assets: list[str] = []
        self.scripts: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        attributes = dict(attrs)
        if "id" in attributes:
            self.ids.add(attributes["id"])
        for key in ("src", "href"):
            target = attributes.get(key)
            if not target or target.startswith(("http://", "https://", "#", "mailto:")):
                continue
            self.local_assets.append(target)
            if tag == "script" and key == "src":
                self.scripts.append(target)


def parse_site() -> SiteParser:
    parser = SiteParser()
    parser.feed((SITE_ROOT / "index.html").read_text(encoding="utf-8"))
    return parser


def test_interactive_site_has_required_labs() -> None:
    parser = parse_site()
    assert {"convolution-lab", "canny-lab", "harris-lab", "matching-lab", "stereo-lab"} <= parser.ids
    assert "driving-map-title" in parser.ids


def test_all_local_site_assets_exist() -> None:
    parser = parse_site()
    missing = [target for target in parser.local_assets if not (SITE_ROOT / target).is_file()]
    assert not missing, f"Missing local site assets: {missing}"


def test_site_uses_local_javascript_without_inline_handlers() -> None:
    html = (SITE_ROOT / "index.html").read_text(encoding="utf-8")
    parser = parse_site()
    assert parser.scripts == ["app.js"]
    assert "onclick=" not in html
    assert "oninput=" not in html


def test_convolution_workbench_is_horizontally_scrollable() -> None:
    html = (SITE_ROOT / "index.html").read_text(encoding="utf-8")
    styles = (SITE_ROOT / "styles.css").read_text(encoding="utf-8")
    assert 'class="matrix-scroll"' in html
    assert 'aria-label="卷积矩阵，可横向滑动"' in html
    assert 'tabindex="0"' in html
    assert 'data-scroll-matrix="-1"' in html
    assert 'data-scroll-matrix="1"' in html
    assert "overflow-x: auto" in styles
    assert "touch-action: pan-x pan-y" in styles
    assert "-webkit-overflow-scrolling: touch" in styles
    assert "scroll-snap-type: x proximity" in styles


def test_convolution_intuition_diagram_is_valid_svg() -> None:
    root = ElementTree.parse(SITE_ROOT / "assets" / "chapter01_convolution_intuition.svg").getroot()
    assert root.tag.endswith("svg")
    assert root.find("{http://www.w3.org/2000/svg}title") is not None


def test_javascript_contains_all_interactions() -> None:
    javascript = (SITE_ROOT / "app.js").read_text(encoding="utf-8")
    for feature in ("calculateConvolution", "cannyStages", "renderRegion", "renderRansacThreshold", "renderStereoDepth", "themeToggle", "setPointerCapture", "scrollLeft", "scrollMatrix", "requestAnimationFrame", 'pointerType !== "mouse"'):
        assert feature in javascript
