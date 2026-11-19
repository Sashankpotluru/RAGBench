import json
import shutil
from hashlib import sha256
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlparse

import pytest

from scripts.build_site import PUBLIC, RELEASE, build

ROOT = Path(__file__).resolve().parents[1]


def release_tree(root):
    report = json.loads((ROOT / RELEASE).read_bytes())
    for name in [*PUBLIC.values(), RELEASE, *(x["path"] for x in report["manifest"])]:
        dest = root / name
        dest.parent.mkdir(exist_ok=True, parents=True)
        shutil.copyfile(ROOT / name, dest)
    return report


class References(HTMLParser):
    def __init__(self):
        super().__init__()
        self.urls = []
        self.ids = set()

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        self.urls.extend(values[k] for k in ("href", "src") if k in values)
        if "id" in values:
            self.ids.add(values["id"])


def test_publish_exactly_verified_inventory_and_project_relative_paths(tmp_path):
    report = release_tree(tmp_path)
    for name in ["site/stale.html", ".superpowers/private", "data/cache/corpus", "web/.env"]:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("unpublished")
    site = build(tmp_path)
    expected = {
        "index.html",
        "style.css",
        "app.mjs",
        "analysis.mjs",
        "LICENSE",
        "THIRD_PARTY_NOTICES.md",
        "docs/architecture.svg",
        "docs/architecture.mmd",
        RELEASE,
        *(row["path"] for row in report["manifest"]),
    }
    catalog = json.loads((site / "site-manifest.json").read_bytes())
    assert {r["path"] for r in catalog["files"]} == expected
    assert {str(p.relative_to(site)) for p in site.rglob("*") if p.is_file()} == expected | {
        "site-manifest.json"
    }
    for row in catalog["files"]:
        data = (site / row["path"]).read_bytes()
        assert len(data) == row["bytes"]
        assert sha256(data).hexdigest() == row["sha256"]
    parser = References()
    parser.feed((site / "index.html").read_text())
    urls = [*parser.urls, report["explorer"], *(r["path"] for r in report["manifest"])]
    for url in urls:
        if url.startswith("#"):
            assert url[1:] in parser.ids
        elif not urlparse(url).scheme:
            resolved = urlparse(urljoin("https://example.org/RAGBench/", url)).path
            assert resolved.startswith("/RAGBench/")
            assert (site / resolved.removeprefix("/RAGBench/")).is_file()


def test_incomplete_release_never_replaces_previous_site(tmp_path):
    release_tree(tmp_path)
    (tmp_path / "results/answers/input-binding.json").unlink()
    (tmp_path / "site").mkdir()
    marker = tmp_path / "site/previous"
    marker.write_text("prior verified build")
    with pytest.raises((ValueError, OSError)):
        build(tmp_path)
    assert marker.read_text() == "prior verified build"
