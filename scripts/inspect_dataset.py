"""
Dataset Inspection Script for LOL (LOw-Light) Dataset
Course: Image and Video Processing (IVP) Mini-Project
"""

import os
from pathlib import Path
from PIL import Image

def get_dir_size(path: Path) -> int:
    """Calculate total size of files in a directory in bytes."""
    total = 0
    for entry in path.rglob('*'):
        if entry.is_file():
            total += entry.stat().st_size
    return total

def inspect_dataset(dataset_root: Path):
    print("=" * 70)
    print("LOL DATASET DETAILED INSPECTION REPORT")
    print("=" * 70)
    
    if not dataset_root.exists():
        print(f"Error: Dataset directory {dataset_root} not found!")
        return

    print(f"Dataset root path: {dataset_root.resolve()}")
    
    
    # 1. Total dataset size
    total_bytes = get_dir_size(dataset_root)
    total_mb = total_bytes / (1024 * 1024)
    print(f"Total Dataset Size: {total_mb:.2f} MB ({total_bytes:,} bytes)")
    
    # 2. Check for metadata / non-image files
    non_image_files = []
    all_files = list(dataset_root.rglob('*'))
    files_only = [f for f in all_files if f.is_file()]
    
    for f in files_only:
        if f.suffix.lower() not in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']:
            non_image_files.append(f)
            
    print(f"\nMetadata / Non-image files found: {len(non_image_files)}")
    for f in non_image_files:
        print(f"  - {f.relative_to(dataset_root)}")
    if not non_image_files:
        print("  (None found. The dataset contains image files organized in directories.)")

    # 3. Explore subdirectories
    subdirs = [p for p in dataset_root.iterdir() if p.is_dir()]
    print(f"\nPrimary Subdirectories: {[d.name for d in subdirs]}")
    
    dataset_summary = {}

    for split_dir in subdirs:
        print(f"\n--- Inspecting subset: '{split_dir.name}' ---")
        split_subdirs = [p for p in split_dir.iterdir() if p.is_dir()]
        print(f"  Subfolders: {[d.name for d in split_subdirs]}")
        
        low_dir = split_dir / 'low'
        high_dir = split_dir / 'high'
        
        has_low = low_dir.exists() and low_dir.is_dir()
        has_high = high_dir.exists() and high_dir.is_dir()
        
        low_files = sorted([f for f in low_dir.glob('*') if f.is_file()]) if has_low else []
        high_files = sorted([f for f in high_dir.glob('*') if f.is_file()]) if has_high else []
        
        print(f"  'low' images count : {len(low_files)}")
        print(f"  'high' images count: {len(high_files)}")
        
        # Check extensions
        low_exts = {f.suffix.lower() for f in low_files}
        high_exts = {f.suffix.lower() for f in high_files}
        print(f"  Extensions in 'low' : {low_exts}")
        print(f"  Extensions in 'high': {high_exts}")
        
        # Check naming conventions and pairings
        low_names = {f.name for f in low_files}
        high_names = {f.name for f in high_files}
        
        common_names = low_names.intersection(high_names)
        only_in_low = low_names - high_names
        only_in_high = high_names - low_names
        
        is_exact_paired = (len(low_files) == len(high_files) == len(common_names))
        print(f"  Exact name 1-to-1 pairing: {is_exact_paired}")
        print(f"  Common filenames count   : {len(common_names)}")
        if only_in_low:
            print(f"  Files only in 'low'      : {len(only_in_low)} (e.g., {list(only_in_low)[:3]})")
        if only_in_high:
            print(f"  Files only in 'high'     : {len(only_in_high)} (e.g., {list(only_in_high)[:3]})")
            
        # Sample filenames
        if low_files:
            print(f"  Sample filenames (low)   : {[f.name for f in low_files[:5]]}")
        if high_files:
            print(f"  Sample filenames (high)  : {[f.name for f in high_files[:5]]}")
            
        # Check image dimensions & channels
        resolutions_low = {}
        modes_low = {}
        for f in low_files:
            try:
                with Image.open(f) as img:
                    res = img.size # (width, height)
                    mode = img.mode
                    resolutions_low[res] = resolutions_low.get(res, 0) + 1
                    modes_low[mode] = modes_low.get(mode, 0) + 1
            except Exception as e:
                print(f"  Error reading {f.name}: {e}")
                
        resolutions_high = {}
        modes_high = {}
        for f in high_files:
            try:
                with Image.open(f) as img:
                    res = img.size
                    mode = img.mode
                    resolutions_high[res] = resolutions_high.get(res, 0) + 1
                    modes_high[mode] = modes_high.get(mode, 0) + 1
            except Exception as e:
                print(f"  Error reading {f.name}: {e}")

        print(f"  Low images dimensions (W x H)  : {resolutions_low}")
        print(f"  Low images color modes         : {modes_low}")
        print(f"  High images dimensions (W x H) : {resolutions_high}")
        print(f"  High images color modes        : {modes_high}")
        
        # Verify resolution match between paired images
        mismatched_dims = []
        for name in list(common_names)[:50]: # check first 50 common
            f_low = low_dir / name
            f_high = high_dir / name
            with Image.open(f_low) as img_l, Image.open(f_high) as img_h:
                if img_l.size != img_h.size:
                    mismatched_dims.append((name, img_l.size, img_h.size))
                    
        if mismatched_dims:
            print(f"  WARNING: Dimension mismatches found in pairs: {mismatched_dims[:3]}")
        else:
            print(f"  Paired image dimensions check: 100% matched across tested pairs.")
            
        dataset_summary[split_dir.name] = {
            "low_count": len(low_files),
            "high_count": len(high_files),
            "exact_paired": is_exact_paired,
            "resolutions": resolutions_low,
            "modes": modes_low
        }

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    total_low = sum(d["low_count"] for d in dataset_summary.values())
    total_high = sum(d["high_count"] for d in dataset_summary.values())
    print(f"Total Low-light Images: {total_low}")
    print(f"Total Reference/Normal Images: {total_high}")
    print(f"Total Image Files: {total_low + total_high}")
    for split, info in dataset_summary.items():
        print(f"  - Split '{split}': {info['low_count']} low-light, {info['high_count']} normal (Paired: {info['exact_paired']})")
    print("=" * 70)

if __name__ == "__main__":
    current_dir = Path(__file__).resolve().parent.parent
    dataset_path = current_dir / "lol_dataset"
    inspect_dataset(dataset_path)
