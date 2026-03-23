#!/usr/bin/env python3
"""Build caption metadata from CellwText LMDBs.

This script reuses ScgptDataset loader fields (celltype/disease/tissue/definitions)
and exports captions as JSONL or JSON mapping by global sample index.
"""

import argparse
import json
import os
import sys
from pathlib import Path


def build_caption(sample: dict, style: str) -> str:
    celltype = str(sample.get("celltype_name", "")).strip()
    disease = str(sample.get("disease_name", "")).strip()
    tissue = str(sample.get("tissue_name", "")).strip()

    celltype_def = str(sample.get("celltype_definition", "")).strip()
    disease_def = str(sample.get("disease_definition", "")).strip()
    tissue_def = str(sample.get("tissue_definition", "")).strip()

    if style == "compact":
        parts = []
        if celltype:
            parts.append(f"This cell is a {celltype}")
        else:
            parts.append("This cell")
        if disease and disease.lower() not in {"nan", "none", "unknown", ""}:
            parts.append(f"under {disease} condition")
        if tissue and tissue.lower() not in {"nan", "none", "unknown", ""}:
            parts.append(f"from {tissue}")
        return " ".join(parts).strip() + "."

    # verbose style (default)
    fields = []
    if celltype:
        fields.append(f"celltype: {celltype}")
    if celltype_def:
        fields.append(f"celltype_definition: {celltype_def}")
    if disease:
        fields.append(f"disease: {disease}")
    if disease_def:
        fields.append(f"disease_definition: {disease_def}")
    if tissue:
        fields.append(f"tissue: {tissue}")
    if tissue_def:
        fields.append(f"tissue_definition: {tissue_def}")
    return "; ".join(fields)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare CellwText captions from LMDB metadata")
    p.add_argument("--cellwtext_dir", default="/data/bgi/data/projects/multimodal/RNA_data/cellwtext_data/CellwText")
    p.add_argument("--lmdb_vocab", default="/data/bgi/data/projects/multimodal/zyh/datasets/CellwText/vocab/gene_vocab.json")
    p.add_argument("--scgpt_vocab", default="/data/bgi/data/projects/multimodal/zyh/datasets/CellwText/scgpt/vocab.json")
    p.add_argument("--celltype_label", default="/data/bgi/data/projects/multimodal/zyh/datasets/CellwText/vocab/celltype_label.json")
    p.add_argument("--out", default="../data_process/cellwtext_captions/captions_verbose.jsonl")
    p.add_argument("--style", choices=["verbose", "compact"], default="verbose")
    p.add_argument("--max_samples", type=int, default=-1, help="-1 means all")
    p.add_argument("--format", choices=["jsonl", "json"], default="jsonl")
    p.add_argument("--log_every", type=int, default=50000)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    parent_dir = str(repo_root.parent)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

    from scgpt.load_CellwText_dataset import ScgptDataset

    dataset = ScgptDataset(
        args.cellwtext_dir,
        lmdb_gene_vocab=args.lmdb_vocab,
        scgpt_gene_vocab=args.scgpt_vocab,
        celltype_label_path=args.celltype_label,
        flag_text_aug=False,
    )

    total = len(dataset)
    limit = total if args.max_samples < 0 else min(total, args.max_samples)

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.format == "jsonl":
        with out_path.open("w", encoding="utf-8") as f:
            for i in range(limit):
                raw = dataset.get_lmdb_data(i)
                item = {
                    "idx": i,
                    "celltype": raw.get("celltype_name", ""),
                    "disease": raw.get("disease_name", ""),
                    "tissue": raw.get("tissue_name", ""),
                    "caption": build_caption(raw, args.style),
                }
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                if (i + 1) % args.log_every == 0:
                    print(f"processed {i + 1}/{limit}")
    else:
        payload = {}
        for i in range(limit):
            raw = dataset.get_lmdb_data(i)
            payload[str(i)] = {
                "celltype": raw.get("celltype_name", ""),
                "disease": raw.get("disease_name", ""),
                "tissue": raw.get("tissue_name", ""),
                "caption": build_caption(raw, args.style),
            }
            if (i + 1) % args.log_every == 0:
                print(f"processed {i + 1}/{limit}")
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)

    print(f"saved: {out_path}")
    print(f"samples: {limit}")


if __name__ == "__main__":
    main()
