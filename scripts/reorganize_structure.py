#!/usr/bin/env python3
"""
Reorganize existing data and artifacts to match the new structure.

New structure:
- data/clean/ (unchanged)
- data/clean_8k/ (was data/clean8k/)
- data/degraded/<dataset_name>/<noise_type>/ (was data/degraded/ flat)
- artifacts/experiments/<dataset_name>/ (already correct)

This script:
1. Moves data/clean8k/ to data/clean_8k/
2. Reorganizes data/degraded/ into noise-type subdirectories
3. Updates manifest files with new paths
4. Handles data_musan subdirectories if present

Usage:
  python reorganize_structure.py --dry-run    # Preview changes
  python reorganize_structure.py --execute   # Apply changes
"""
import argparse
import csv
import shutil
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def get_noise_type_from_filename(filename):
    """Extract noise type from degraded filename."""
    # Filename format: hi_XXXXXX__<noise_type>_snrX_brY_cZ.wav or hi_XXXXXX__none_brY.wav
    parts = filename.split("__")
    if len(parts) < 2:
        return "unknown"
    noise_part = parts[1].split("_")[0]  # Get first part after __
    return noise_part


def reorganize_degraded_files(degraded_dir, dry_run=True):
    """Reorganize degraded files into noise-type subdirectories."""
    degraded_dir = Path(degraded_dir)
    if not degraded_dir.exists():
        print(f"Degraded directory not found: {degraded_dir}")
        return
    
    # Determine dataset name from parent directory
    dataset_name = degraded_dir.name if degraded_dir.name != "degraded" else "synthetic"
    if degraded_dir.parent.name == "data":
        # If it's data/degraded/, use "synthetic"
        dataset_name = "synthetic"
    
    # Group files by noise type
    files_by_noise = defaultdict(list)
    for wav_file in degraded_dir.glob("*.wav"):
        noise_type = get_noise_type_from_filename(wav_file.name)
        files_by_noise[noise_type].append(wav_file)
    
    print(f"\nFound {len(files_by_noise)} noise types in {degraded_dir}:")
    for noise_type, files in sorted(files_by_noise.items()):
        print(f"  {noise_type}: {len(files)} files")
    
    if dry_run:
        print("\nDRY RUN: Would create directories and move files")
        for noise_type, files in sorted(files_by_noise.items()):
            target_dir = degraded_dir / dataset_name / noise_type
            print(f"  mkdir -p {target_dir}")
            for f in files[:3]:  # Show first 3
                print(f"    mv {f.name} -> {target_dir / f.name}")
            if len(files) > 3:
                print(f"    ... and {len(files) - 3} more files")
    else:
        # Create directories and move files
        for noise_type, files in sorted(files_by_noise.items()):
            target_dir = degraded_dir / dataset_name / noise_type
            target_dir.mkdir(parents=True, exist_ok=True)
            print(f"Created {target_dir}")
            
            for f in files:
                target = target_dir / f.name
                if target.exists():
                    print(f"  Skipping {f.name} (already exists at target)")
                    continue
                shutil.move(str(f), str(target))
            print(f"  Moved {len(files)} files to {target_dir}")


def rename_clean8k_to_clean_8k(data_dir, dry_run=True):
    """Rename data/clean8k/ to data/clean_8k/."""
    clean8k_dir = data_dir / "clean8k"
    clean_8k_dir = data_dir / "clean_8k"
    
    if not clean8k_dir.exists():
        print(f"clean8k directory not found: {clean8k_dir}")
        return
    
    if clean_8k_dir.exists():
        print(f"clean_8k already exists: {clean_8k_dir}")
        return
    
    if dry_run:
        print(f"\nDRY RUN: Would rename {clean8k_dir} -> {clean_8k_dir}")
    else:
        shutil.move(str(clean8k_dir), str(clean_8k_dir))
        print(f"Renamed {clean8k_dir} -> {clean_8k_dir}")


def update_manifest_paths(data_dir, dry_run=True):
    """Update manifest files with new paths."""
    manifest_path = data_dir / "manifest.csv"
    degraded_manifest_path = data_dir / "manifest_degraded.csv"
    
    # Update manifest.csv (clean8k paths)
    if manifest_path.exists():
        if dry_run:
            print(f"\nDRY RUN: Would update {manifest_path} (clean8k -> clean_8k)")
        else:
            rows = []
            with open(manifest_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("clean8k_path"):
                        row["clean8k_path"] = row["clean8k_path"].replace("clean8k/", "clean_8k/")
                    rows.append(row)
            
            with open(manifest_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            print(f"Updated {manifest_path}")
    
    # Update manifest_degraded.csv (degraded paths and clean8k paths)
    if degraded_manifest_path.exists():
        if dry_run:
            print(f"\nDRY RUN: Would update {degraded_manifest_path} (degraded paths and clean8k -> clean_8k)")
        else:
            rows = []
            with open(degraded_manifest_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Update clean8k_path
                    if row.get("clean8k_path"):
                        row["clean8k_path"] = row["clean8k_path"].replace("clean8k/", "clean_8k/")
                    # Update degraded_path
                    if row.get("degraded_path"):
                        # Extract noise type from filename and update path
                        deg_id = row["deg_id"]
                        noise_type = get_noise_type_from_filename(deg_id)
                        dataset_name = "synthetic"  # Default for data/degraded/
                        # Check if path already has dataset name
                        if "/" in row["degraded_path"] and row["degraded_path"].split("/")[0] == "degraded":
                            parts = row["degraded_path"].split("/")
                            if len(parts) > 2 and parts[1] != dataset_name:
                                dataset_name = parts[1]
                        new_path = f"degraded/{dataset_name}/{noise_type}/{deg_id}.wav"
                        row["degraded_path"] = new_path
                    rows.append(row)
            
            with open(degraded_manifest_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            print(f"Updated {degraded_manifest_path}")


def handle_dataset_subdir(data_dir, subdir_name, dry_run=True):
    """Handle a dataset subdirectory (e.g., data_musan, data_demand) if present."""
    dataset_dir = data_dir / subdir_name
    if not dataset_dir.exists():
        return
    
    print(f"\nFound {subdir_name} directory: {dataset_dir}")
    
    # Check if it has clean8k
    clean8k_dir = dataset_dir / "clean8k"
    clean_8k_dir = dataset_dir / "clean_8k"
    if clean8k_dir.exists() and not clean_8k_dir.exists():
        if dry_run:
            print(f"DRY RUN: Would rename {clean8k_dir} -> {clean_8k_dir}")
        else:
            shutil.move(str(clean8k_dir), str(clean_8k_dir))
            print(f"Renamed {clean8k_dir} -> {clean_8k_dir}")
    
    # Check if it has degraded directory
    degraded_dir = dataset_dir / "degraded"
    if degraded_dir.exists():
        reorganize_degraded_files(degraded_dir, dry_run)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="Preview changes without executing")
    ap.add_argument("--execute", action="store_true", help="Actually perform the reorganization")
    args = ap.parse_args()
    
    if not args.dry_run and not args.execute:
        print("Please specify either --dry-run or --execute")
        return
    
    dry_run = args.dry_run
    
    print("=" * 60)
    print("REORGANIZATION SCRIPT")
    print("=" * 60)
    print(f"Mode: {'DRY RUN' if dry_run else 'EXECUTE'}")
    print(f"Data directory: {DATA_DIR}")
    
    # Step 1: Rename clean8k to clean_8k
    print("\n" + "=" * 60)
    print("STEP 1: Rename clean8k -> clean_8k")
    print("=" * 60)
    rename_clean8k_to_clean_8k(DATA_DIR, dry_run)
    
    # Step 2: Reorganize degraded files
    print("\n" + "=" * 60)
    print("STEP 2: Reorganize degraded files")
    print("=" * 60)
    degraded_dir = DATA_DIR / "degraded"
    reorganize_degraded_files(degraded_dir, dry_run)
    
    # Step 3: Handle dataset subdirectories (data_musan, data_demand)
    print("\n" + "=" * 60)
    print("STEP 3: Handle dataset subdirectories")
    print("=" * 60)
    handle_dataset_subdir(DATA_DIR, "data_musan", dry_run)
    handle_dataset_subdir(DATA_DIR, "data_demand", dry_run)
    
    # Step 4: Update manifest files
    print("\n" + "=" * 60)
    print("STEP 4: Update manifest files")
    print("=" * 60)
    update_manifest_paths(DATA_DIR, dry_run)
    update_manifest_paths(DATA_DIR / "data_musan", dry_run)
    update_manifest_paths(DATA_DIR / "data_demand", dry_run)
    
    print("\n" + "=" * 60)
    if dry_run:
        print("DRY RUN COMPLETE. Review the output above.")
        print("Run with --execute to apply changes.")
    else:
        print("REORGANIZATION COMPLETE.")
    print("=" * 60)


if __name__ == "__main__":
    main()
