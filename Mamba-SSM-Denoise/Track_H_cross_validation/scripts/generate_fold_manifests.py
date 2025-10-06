#!/usr/bin/env python3
"""Generate train/val/test manifests for a specific cross-validation fold."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

ManifestRow = Dict[str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, required=True, help="Fold index (1-based)")
    parser.add_argument(
        "--config",
        default="configs/folds.json",
        help="Path to the folds configuration file (default: configs/folds.json)",
    )
    parser.add_argument(
        "--input-manifest",
        default="data/spectra_for_fold/manifest.csv",
        help="Path to the full dataset manifest",
    )
    parser.add_argument(
        "--output-root",
        default="manifests",
        help="Directory where fold manifests will be written",
    )
    return parser.parse_args()


def load_config(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_manifest(path: Path) -> List[ManifestRow]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        raise ValueError(f"Manifest {path} is empty")
    return rows


def build_assignments(config: Dict, fold: int) -> Dict[str, Dict[str, List[str] | str]]:
    fold_key = str(fold)
    if fold_key not in config["folds"]:
        raise KeyError(f"Fold {fold} is not defined in {config}")
    fold_spec = config["folds"][fold_key]
    assignments: Dict[str, Dict[str, List[str] | str]] = {}
    for treatment, train_samples in fold_spec["train"].items():
        val_sample = fold_spec["val"].get(treatment)
        test_sample = fold_spec["test"].get(treatment)
        if val_sample is None or test_sample is None:
            raise KeyError(f"Fold {fold} missing val/test sample for {treatment}")
        assignments[treatment] = {
            "train": set(train_samples),
            "val": {val_sample},
            "test": {test_sample},
        }
    return assignments


def select_rows(rows: List[ManifestRow], assignments: Dict[str, Dict[str, set]]) -> Dict[str, List[ManifestRow]]:
    buckets = {"train": [], "val": [], "test": []}
    for row in rows:
        treatment = row["treatment"]
        sample = row["sample"]
        if treatment not in assignments:
            continue
        role_sets = assignments[treatment]
        if sample in role_sets["train"]:
            buckets["train"].append(row)
        elif sample in role_sets["val"]:
            buckets["val"].append(row)
        elif sample in role_sets["test"]:
            buckets["test"].append(row)
    for role, bucket_rows in buckets.items():
        if not bucket_rows:
            raise ValueError(f"No rows selected for {role}; check fold configuration")
    return buckets


def write_manifest(path: Path, header: List[str], rows: List[ManifestRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    config = load_config(Path(args.config))
    rows = load_manifest(Path(args.input_manifest))
    header = list(rows[0].keys())

    assignments = build_assignments(config, args.fold)
    buckets = select_rows(rows, assignments)

    output_root = Path(args.output_root) / f"fold_{args.fold:02d}"
    for role, role_rows in buckets.items():
        write_manifest(output_root / f"{role}_manifest.csv", header, role_rows)

    print(f"Generated manifests in {output_root}")


if __name__ == "__main__":
    main()
