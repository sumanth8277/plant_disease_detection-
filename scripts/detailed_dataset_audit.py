"""
Comprehensive Dataset & Class Audit Script (scripts/detailed_dataset_audit.py)
------------------------------------------------------------------------------
Audits all 38 canonical classes in dataset/raw/PlantVillage:
- File counts per folder
- PIL image integrity (corruption detection)
- File format & dimension distribution
- Exact duplicates via SHA-256
- Near duplicates via dHash (difference hash)
- Conflicting labels / cross-folder duplicates
- Identifies empty folders and low-sample classes
"""

import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from config import Config

def compute_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def compute_dhash(img: Image.Image, hash_size: int = 8) -> str:
    try:
        resized = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
        pixels = list(resized.tobytes())
        diff = []
        for row in range(hash_size):
            for col in range(hash_size):
                idx = row * (hash_size + 1) + col
                diff.append("1" if pixels[idx] > pixels[idx + 1] else "0")
        return hex(int("".join(diff), 2))[2:].zfill((hash_size * hash_size) // 4)
    except Exception:
        return ""

def audit_dataset():
    pv_dir = BASE_DIR / "dataset" / "raw" / "PlantVillage"
    classes_file = BASE_DIR / "models" / "class_names.json"
    
    with open(classes_file, "r", encoding="utf-8") as f:
        canonical_classes = json.load(f)
    
    results = {
        "dataset_root": str(pv_dir),
        "total_classes_checked": len(canonical_classes),
        "classes": {},
        "summary": {
            "total_images": 0,
            "total_corrupted": 0,
            "classes_with_images": 0,
            "empty_classes": 0,
            "classes_under_10_images": 0,
            "exact_duplicate_pairs": [],
            "near_duplicate_pairs": [],
            "cross_class_duplicates": []
        }
    }
    
    sha256_to_files = defaultdict(list)
    dhash_to_files = defaultdict(list)
    
    for cls_name in canonical_classes:
        cls_dir = pv_dir / cls_name
        cls_info = {
            "exists": cls_dir.exists(),
            "valid_count": 0,
            "corrupt_count": 0,
            "corrupt_files": [],
            "dimensions": defaultdict(int),
            "formats": defaultdict(int),
            "files": []
        }
        
        if cls_dir.exists() and cls_dir.is_dir():
            for entry in cls_dir.iterdir():
                if entry.is_file() and entry.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]:
                    try:
                        with Image.open(entry) as img:
                            img.verify()
                        with Image.open(entry) as img:
                            dim = f"{img.width}x{img.height}"
                            fmt = img.format
                            cls_info["dimensions"][dim] += 1
                            cls_info["formats"][fmt] += 1
                            dh = compute_dhash(img)
                        
                        sha = compute_sha256(entry)
                        cls_info["valid_count"] += 1
                        cls_info["files"].append(entry.name)
                        sha256_to_files[sha].append((cls_name, str(entry)))
                        if dh:
                            dhash_to_files[dh].append((cls_name, str(entry)))
                            
                    except Exception as e:
                        cls_info["corrupt_count"] += 1
                        cls_info["corrupt_files"].append({"file": entry.name, "error": str(e)})
        
        cls_info["dimensions"] = dict(cls_info["dimensions"])
        cls_info["formats"] = dict(cls_info["formats"])
        results["classes"][cls_name] = cls_info
        
        results["summary"]["total_images"] += cls_info["valid_count"]
        results["summary"]["total_corrupted"] += cls_info["corrupt_count"]
        if cls_info["valid_count"] > 0:
            results["summary"]["classes_with_images"] += 1
            if cls_info["valid_count"] < 10:
                results["summary"]["classes_under_10_images"] += 1
        else:
            results["summary"]["empty_classes"] += 1

    # Duplicates check
    for sha, flist in sha256_to_files.items():
        if len(flist) > 1:
            classes_involved = set(c for c, _ in flist)
            results["summary"]["exact_duplicate_pairs"].append({
                "sha256": sha,
                "occurrences": flist
            })
            if len(classes_involved) > 1:
                results["summary"]["cross_class_duplicates"].append({
                    "sha256": sha,
                    "classes": list(classes_involved),
                    "files": flist
                })

    for dh, flist in dhash_to_files.items():
        if len(flist) > 1:
            results["summary"]["near_duplicate_pairs"].append({
                "dhash": dh,
                "occurrences": flist
            })

    out_file = BASE_DIR / "reports" / "detailed_class_audit.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Audit completed: {results['summary']['total_images']} valid images found.")
    print(f"Populated classes: {results['summary']['classes_with_images']}/{len(canonical_classes)}")
    print(f"Empty classes: {results['summary']['empty_classes']}/{len(canonical_classes)}")
    print(f"Corrupt images: {results['summary']['total_corrupted']}")
    print(f"Exact duplicates: {len(results['summary']['exact_duplicate_pairs'])}")
    print(f"Cross-class duplicates: {len(results['summary']['cross_class_duplicates'])}")
    print(f"Detailed audit saved to: {out_file}")

if __name__ == "__main__":
    audit_dataset()
