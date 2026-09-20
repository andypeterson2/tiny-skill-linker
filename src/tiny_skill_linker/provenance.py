"""What produced a run: the code it ran and the weights it loaded."""

import hashlib
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
CODE_DIRS = ("scripts", "src")


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def code_digest(root: Path = ROOT) -> str:
    """Digest every .py file under scripts/ and src/, path and contents.

    A result stays checkable when the commit it names is unreachable.
    """
    digest = hashlib.sha256()
    for path in sorted(p for d in CODE_DIRS for p in (root / d).rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(file_digest(path).encode())
    return digest.hexdigest()


def code_state(root: Path = ROOT) -> dict:
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
    )
    # Scoped to the code. An uncommitted result from an earlier run in the same
    # batch is not a modified script.
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--", *CODE_DIRS, "pyproject.toml"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    return {
        "sha": sha.stdout.strip(),
        "dirty": bool(dirty.stdout.strip()),
        "code_digest": code_digest(root),
    }


def weights_digest(model: str, onnx_file: str | None = None) -> str | None:
    """Digest of the weight file that is loaded, or None for a hub model."""
    root = Path(model)
    if not root.exists():
        return None
    target = root / onnx_file if onnx_file else root / "model.safetensors"
    return file_digest(target) if target.exists() else None
