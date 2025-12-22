"""
Manifest loading helpers.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from knowledge.models import CorpusManifest


def load_corpus_manifest(manifest_path: str | Path) -> CorpusManifest:
    path = Path(manifest_path)
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return CorpusManifest.model_validate(data)


