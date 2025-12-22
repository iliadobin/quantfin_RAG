"""
Read/write helpers for ingest artifacts under data/parsed/.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Union

from pydantic import BaseModel


JsonlItem = Union[BaseModel, dict]


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_jsonl(path: str | Path, items: Iterable[JsonlItem]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for it in items:
            if isinstance(it, BaseModel):
                obj = it.model_dump(mode="json")
            else:
                obj = it
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


