#!/usr/bin/env python3
"""
Comprehensive reorganization of data and artifacts to match the new structure.

Target structure:
- data/clean/ (unchanged)
- data/clean_8k/ (unchanged)
- data/degraded/synthetic/<noise_type>/ (unchanged)
- data/data_demand/clean_8k/ (moved from data/noise/data_demand/clean8k)
- data/data_demand/degraded/<noise_type>/ (moved from data/datasets/data_demand/degraded)
- data/data_musan/clean_8k/ (unchanged)
- data/data_musan/degraded/<noise_type>/ (moved from data/datasets/data_musan/degraded)
- data/noise/demand/<noise_type>/ (noise files only, unchanged)
- artifacts/experiments/data_demand/ (consolidated)
- artifacts/experiments/data_musan/ (unchanged)

Usage:
  python reorganize_comprehensive.py --dry-run    # Preview changes
  python reorganize_comprehensive.py --execute   # Apply changes
"""
import argparse
import csv
import shutil
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


def get_noise_type_from_filename(filename):
    """Extract noise type from degraded filename."""
    # Filename format: hi_XXXXXX__<noise_type>_snrX_brY_cZ.wav or hi_XXXXXX__none_brY.wav
    parts = filename.split("__")
    if len(parts) < 2:
        return "unknown"
    noise_part = parts[1].split("_")[0]  # Get first part after __
    return noise_part


def reorganize_degraded_files(degraded_dir, dataset_name, target_base_dir, dry_run=True):
    """Reorganize degraded files into noise-type subdirectories."""
    degraded_dir = Path(degraded_dir)
    if not degraded_dir.exists():
        print(f"Degraded directory not found: {degraded_dir}")
        return
    
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
            target_dir = target_base_dir / dataset_name / noise_type
            print(f"  mkdir -p {target_dir}")
            for f in files[:3]:  # Show first 3
                print(f"    mv {f.name} -> {target_dir / f.name}")
            if len(files) > 3:
                print(f"    ... and {len(files) - 3} more files")
    else:
        # Create directories and move files
        for noise_type, files in sorted(files_by_noise.items()):
            target_dir = target_base_dir / dataset_name / noise_type
            target_dir.mkdir(parents=True, exist_ok=True)
            print(f"Created {target_dir}")
            
            for f in files:
                target = target_dir / f.name
                if target.exists():
                    print(f"  Skipping {f.name} (already exists at target)")
                    continue
                shutil.move(str(f), str(target))
            print(f"  Moved {len(files)} files to {target_dir}")
        
        # Remove old degraded directory if empty
        if degraded_dir.exists() and not any(degraded_dir.iterdir()):
            degraded_dir.rmdir()
            print(f"Removed empty directory: {degraded_dir}")


def move_directory(src, dst, dry_run=True):
    """Move a directory from src to dst."""
    src = Path(src)
    dst = Path(dst)
    
    if not src.exists():
        print(f"Source not found: {src}")
        return
    
    if dst.exists():
        print(f"Destination already exists: {dst}")
        return
    
    if dry_run:
        print(f"DRY RUN: Would move {src} -> {dst}")
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        print(f"Moved {src} -> {dst}")


def update_manifest_degraded_paths(manifest_path, dataset_name, dry_run=True):
    """Update degraded paths and clean8k paths in manifest file."""
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        print(f"Manifest not found: {manifest_path}")
        return
    
    if dry_run:
        print(f"\nDRY RUN: Would update {manifest_path} (degraded paths and clean8k -> clean_8k)")
        return
    
    rows = []
    with open(manifest_path, newline="", encoding="utf-8") as f:
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
                new_path = f"degraded/{noise_type}/{deg_id}.wav"
                row["degraded_path"] = new_path
            rows.append(row)
    
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Updated {manifest_path}")


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
    print("COMPREHENSIVE REORGANIZATION SCRIPT")
    print("=" * 60)
    print(f"Mode: {'DRY RUN' if dry_run else 'EXECUTE'}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Artifacts directory: {ARTIFACTS_DIR}")
    
    # Step 1: Handle data_demand
    print("\n" + "=" * 60)
    print("STEP 1: Reorganize data_demand")
    print("=" * 60)
    
    # Move degraded from datasets/data_demand to data_demand
    datasets_demand_degraded = DATA_DIR / "datasets" / "data_demand" / "degraded"
    data_demand_dir = DATA_DIR / "data_demand"
    if datasets_demand_degraded.exists():
        print(f"\nFound degraded in datasets/data_demand")
        if dry_run:
            print(f"DRY RUN: Would reorganize {datasets_demand_degraded}")
        reorganize_degraded_files(datasets_demand_degraded, "data_demand", DATA_DIR, dry_run)
    
    # Move clean8k from noise/data_demand to data_demand
    noise_demand_clean8k = DATA_DIR / "noise" / "data_demand" / "clean8k"
    data_demand_clean8k = DATA_DIR / "data_demand" / "clean_8k"
    if noise_demand_clean8k.exists() and not data_demand_clean8k.exists():
        move_directory(noise_demand_clean8k, data_demand_clean8k, dry_run)
    
    # Move artifacts from noise/data_demand to artifacts/experiments/data_demand
    noise_demand_artifacts = DATA_DIR / "noise" / "data_demand" / "artifacts"
    artifacts_demand = ARTIFACTS_DIR / "experiments" / "data_demand" / "artifacts"
    if noise_demand_artifacts.exists():
        move_directory(noise_demand_artifacts, artifacts_demand, dry_run)
    
    # Move enh from noise/data_demand to artifacts/experiments/data_demand
    noise_demand_enh = DATA_DIR / "noise" / "data_demand" / "enh"
    artifacts_demand_enh = ARTIFACTS_DIR / "experiments" / "data_demand" / "enh"
    if noise_demand_enh.exists():
        move_directory(noise_demand_enh, artifacts_demand_enh, dry_run)
    
    # Update manifest for data_demand
    datasets_demand_manifest = DATA_DIR / "datasets" / "data_demand" / "manifest_degraded.csv"
    data_demand_manifest = DATA_DIR / "data_demand" / "manifest_degraded.csv"
    if datasets_demand_manifest.exists() and not data_demand_manifest.exists():
        move_directory(datasets_demand_manifest, data_demand_manifest, dry_run)
        if not dry_run:
            update_manifest_degraded_paths(data_demand_manifest, "data_demand", dry_run)
    
    # Step 2: Handle data_musan
    print("\n" + "=" * 60)
    print("STEP 2: Reorganize data_musan")
    print("=" * 60)
    
    # Move degraded from datasets/data_musan to data_musan
    datasets_musan_degraded = DATA_DIR / "datasets" / "data_musan" / "degraded"
    data_musan_dir = DATA_DIR / "data_musan"
    if datasets_musan_degraded.exists():
        print(f"\nFound degraded in datasets/data_musan")
        if dry_run:
            print(f"DRY RUN: Would reorganize {datasets_musan_degraded}")
        reorganize_degraded_files(datasets_musan_degraded, "data_musan", DATA_DIR, dry_run)
    
    # Update manifest for data_musan
    datasets_musan_manifest = DATA_DIR / "datasets" / "data_musan" / "manifest_degraded.csv"
    data_musan_manifest = DATA_DIR / "data_musan" / "manifest_degraded.csv"
    if datasets_musan_manifest.exists() and not data_musan_manifest.exists():
        move_directory(datasets_musan_manifest, data_musan_manifest, dry_run)
        if not dry_run:
            update_manifest_degraded_paths(data_musan_manifest, "data_musan", dry_run)
    
    # Step 3: Clean up empty datasets directory
    print("\n" + "=" * 60)
    print("STEP 3: Clean up empty datasets directory")
    print("=" * 60)
    datasets_dir = DATA_DIR / "datasets"
    if datasets_dir.exists():
        datasets_data_demand = datasets_dir / "data_demand"
        datasets_data_musan = datasets_dir / "data_musan"
        
        if datasets_data_demand.exists() and not any(datasets_data_demand.iterdir()):
            if dry_run:
                print(f"DRY RUN: Would remove empty {datasets_data_demand}")
            else:
                datasets_data_demand.rmdir()
                print(f"Removed empty directory: {datasets_data_demand}")
        
        if datasets_data_musan.exists() and not any(datasets_data_musan.iterdir()):
            if dry_run:
                print(f"DRY RUN: Would remove empty {datasets_data_musan}")
            else:
                datasets_data_musan.rmdir()
                print(f"Removed empty directory: {datasets_data_musan}")
        
        if datasets_dir.exists() and not any(datasets_dir.iterdir()):
            if dry_run:
                print(f"DRY RUN: Would remove empty {datasets_dir}")
            else:
                datasets_dir.rmdir()
                print(f"Removed empty directory: {datasets_dir}")
    
    # Step 4: Clean up empty noise/data_demand directory
    print("\n" + "=" * 60)
    print("STEP 4: Clean up empty noise/data_demand directory")
    print("=" * 60)
    noise_data_demand = DATA_DIR / "noise" / "data_demand"
    if noise_data_demand.exists() and not any(noise_data_demand.iterdir()):
        if dry_run:
            print(f"DRY RUN: Would remove empty {noise_data_demand}")
        else:
            noise_data_demand.rmdir()
            print(f"Removed empty directory: {noise_data_demand}")
    
    print("\n" + "=" * 60)
    if dry_run:
        print("DRY RUN COMPLETE. Review the output above.")
        print("Run with --execute to apply changes.")
    else:
        print("REORGANIZATION COMPLETE.")
    print("=" * 60)


if __name__ == "__main__":
    main()
