"""Package the versioned, replay-verified evidence and the static notebook."""

import json
import sys
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.verify import gate_exit, verify  # noqa: E402

# Explicit site files. The verifier independently reconstructs the remaining
# evidence allowlist from the protocol, including all declared upstream inputs.
PUBLIC = {
    "index.html": "web/index.html",
    "app.mjs": "web/app.mjs",
    "analysis.mjs": "web/analysis.mjs",
    "style.css": "web/style.css",
    "docs/architecture.svg": "docs/architecture.svg",
    "docs/architecture.mmd": "docs/architecture.mmd",
    "LICENSE": "LICENSE",
    "THIRD_PARTY_NOTICES.md": "THIRD_PARTY_NOTICES.md",
}
RELEASE = "results/release.json"


def build(root=ROOT):
    result = root / RELEASE
    raw = result.read_bytes()
    replayed = verify(root, result.parent)
    if gate_exit(replayed["gates"][0]):
        raise ValueError("declared release gate did not pass")
    report = json.loads(raw)
    if result.read_bytes() != raw:
        raise ValueError("release changed during verification")
    paths = dict(PUBLIC)
    paths[RELEASE] = RELEASE
    evidence = {entry["path"]: entry for entry in report["manifest"]}
    if evidence.keys() & paths.keys():
        raise ValueError("evidence and site paths overlap")
    paths.update({name: name for name in evidence})
    target = root / "site"
    if target.is_symlink() or (target.exists() and not target.is_dir()):
        raise ValueError("site output must be a directory")
    with TemporaryDirectory(dir=root, prefix=".site-build-") as tmp:
        stage = Path(tmp) / "site"
        stage.mkdir()
        descriptors = []
        for name, source in sorted(paths.items()):
            src = root / source
            if src.resolve() != src or not src.is_file():
                raise ValueError(f"invalid public source {source}")
            data = raw if name == RELEASE else src.read_bytes()
            entry = {"path": name, "bytes": len(data), "sha256": sha256(data).hexdigest()}
            if name in evidence and entry != evidence[name]:
                raise ValueError(f"evidence changed during packaging: {name}")
            dest = stage / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            descriptors.append(entry)
        manifest = {"schema_version": 1, "files": descriptors}
        (stage / "site-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        # Replace the complete output tree; stale files never enter a fresh deploy.
        prior = Path(tmp) / "prior"
        if target.exists():
            target.rename(prior)
        try:
            stage.rename(target)
        except OSError:
            if prior.exists():
                prior.rename(target)
            raise
    return target


if __name__ == "__main__":
    print(f"Built verified notebook at {build()}")
