#!/usr/bin/env python3
"""
Create empty "production" dataset templates (DS1–DS5) under benchmarks/datasets/production/.

These templates are meant to be filled with REAL doc_id / chunk_id values from:
  - configs/corpus_public.yaml
  - data/indices/public/<strategy>/chunks.jsonl
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from benchmarks.datasets.generator import DatasetGenerator  # noqa: E402
from benchmarks.datasets.loader import save_dataset  # noqa: E402


def main() -> int:
    out_dir = project_root / "benchmarks" / "datasets" / "production"
    out_dir.mkdir(parents=True, exist_ok=True)

    gen = DatasetGenerator(corpus_profile="public")

    ds1 = gen.create_empty_ds1(description="Production DS1 for public corpus (fill with real citations)")
    ds2 = gen.create_empty_ds2(description="Production DS2 qrels for public corpus (fill with real chunk_id)")
    ds3 = gen.create_empty_ds3(description="Production DS3 unanswerable/traps for public corpus")
    ds4 = gen.create_empty_ds4(description="Production DS4 multi-hop for public corpus (fill with real citations)")
    ds5 = gen.create_empty_ds5(description="Production DS5 structured extraction for public corpus (fill with real schema/output)")

    save_dataset(ds1, out_dir / "ds1_factual_qa_public.json")
    save_dataset(ds2, out_dir / "ds2_retrieval_qrels_public.json")
    save_dataset(ds3, out_dir / "ds3_unanswerable_traps_public.json")
    save_dataset(ds4, out_dir / "ds4_multihop_public.json")
    save_dataset(ds5, out_dir / "ds5_structured_extraction_public.json")

    print("✓ Created production dataset templates:")
    for p in [
        "ds1_factual_qa_public.json",
        "ds2_retrieval_qrels_public.json",
        "ds3_unanswerable_traps_public.json",
        "ds4_multihop_public.json",
        "ds5_structured_extraction_public.json",
    ]:
        print(f"  - {out_dir / p}")
    print("\nNext: fill these files with real doc_id/chunk_id values (see benchmarks/datasets/production/README.md).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


