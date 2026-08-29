from pathlib import Path
import re

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MARKDOWN_LINK = re.compile(r"!?\[[^]]*]\(([^)]+)\)")


@pytest.mark.parametrize(
    ("english", "chinese"),
    [
        ("README.md", "README.zh-CN.md"),
        ("docs/roadmap.md", "docs/roadmap.zh-CN.md"),
        ("docs/CASE_STANDARD.md", "docs/CASE_STANDARD.zh-CN.md"),
        ("chapters/01_image_foundations/README.md", "chapters/01_image_foundations/README.zh-CN.md"),
        ("chapters/02_edge_detection/README.md", "chapters/02_edge_detection/README.zh-CN.md"),
        ("chapters/03_harris_corners/README.md", "chapters/03_harris_corners/README.zh-CN.md"),
        ("chapters/04_feature_matching/README.md", "chapters/04_feature_matching/README.zh-CN.md"),
        ("chapters/05_multiview_geometry/README.md", "chapters/05_multiview_geometry/README.zh-CN.md"),
        ("chapters/06_hough_morphology/README.md", "chapters/06_hough_morphology/README.zh-CN.md"),
        ("chapters/07_optical_flow/README.md", "chapters/07_optical_flow/README.zh-CN.md"),
        ("chapters/08_camera_calibration/README.md", "chapters/08_camera_calibration/README.zh-CN.md"),
        ("chapters/09_pnp_pose/README.md", "chapters/09_pnp_pose/README.zh-CN.md"),
        ("chapters/special_convolution/README.md", "chapters/special_convolution/README.zh-CN.md"),
    ],
)
def test_bilingual_document_pairs_exist(english: str, chinese: str) -> None:
    assert (REPOSITORY_ROOT / english).is_file()
    assert (REPOSITORY_ROOT / chinese).is_file()


def test_all_local_markdown_links_resolve() -> None:
    failures: list[str] = []
    for document in REPOSITORY_ROOT.rglob("*.md"):
        if any(part.startswith(".") for part in document.relative_to(REPOSITORY_ROOT).parts):
            continue
        text = document.read_text(encoding="utf-8")
        for target in MARKDOWN_LINK.findall(text):
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            path_text = target.split("#", maxsplit=1)[0]
            resolved = (document.parent / path_text).resolve()
            if not resolved.exists():
                failures.append(f"{document.relative_to(REPOSITORY_ROOT)} -> {target}")

    assert not failures, "Broken local documentation links:\n" + "\n".join(failures)
