from __future__ import annotations

import uuid
from pathlib import Path


def get_id(n: int = 6):
    uuid4 = uuid.uuid4().hex[:n]
    return str(uuid4).replace("-", "")


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_storage_dir(storage_dir: str | Path | None = None) -> Path:
    if storage_dir is None:
        return get_repo_root() / "storage"

    candidate = Path(storage_dir).expanduser()
    if candidate.is_absolute():
        return candidate
    return get_repo_root() / candidate


def iter_document_paths(storage_dir: str | Path | None = None) -> list[Path]:
    root = get_storage_dir(storage_dir)
    if not root.exists():
        return []

    document_paths = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".pdf", ".docx"}
    ]
    return sorted(document_paths, key=lambda path: (path.name.lower(), str(path).lower()))


__all__ = [
    "get_id",
    "get_repo_root",
    "get_storage_dir",
    "iter_document_paths",
]
