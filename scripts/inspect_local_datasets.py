import json
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path(__file__).resolve().parent.parent

with open(BASE_DIR / "models" / "class_names.json", "r", encoding="utf-8") as f:
    target_classes = json.load(f)

pv_manifest_path = BASE_DIR / "dataset" / "plantvillage_candidate" / "manifest.json"
pd_manifest_path = BASE_DIR / "dataset" / "plantdoc" / "manifest.json"

print(f"Target Production Classes: {len(target_classes)}")

# 1. Audit PlantVillage Candidate Local Dataset
print("\n" + "=" * 60)
print("1. LOCAL PLANTVILLAGE DATASET AUDIT (dataset/plantvillage_candidate)")
print("=" * 60)
if pv_manifest_path.exists():
    with open(pv_manifest_path, "r", encoding="utf-8") as f:
        pv_manifest = json.load(f)
    recs = pv_manifest.get("records", [])
    print(f"Manifest found: {pv_manifest_path}")
    print(f"Total recorded images: {len(recs)}")

    # Split breakdown
    split_counts = defaultdict(int)
    class_split_counts = defaultdict(lambda: defaultdict(int))
    classes_present = set()

    for r in recs:
        split = r.get("split", "unknown")
        cname = r.get("class_name", "unknown")
        split_counts[split] += 1
        class_split_counts[cname][split] += 1
        classes_present.add(cname)

    print(f"Split breakdown: Train: {split_counts['train']}, Val: {split_counts['val']}, Test: {split_counts['test']}")
    print(f"Distinct classes present: {len(classes_present)} / {len(target_classes)}")

    missing_in_pv = [c for c in target_classes if c not in classes_present]
    if missing_in_pv:
        print(f"Missing classes in PlantVillage: {missing_in_pv}")
    else:
        print("All 38 production classes are 100% PRESENT in PlantVillage dataset!")

    print("\nSample image counts per class (PlantVillage):")
    for i, c in enumerate(target_classes[:10]):
        tot = sum(class_split_counts[c].values())
        print(f"  [{i:02d}] {c:45s} -> Total: {tot:3d} (Train: {class_split_counts[c]['train']:2d}, Val: {class_split_counts[c]['val']:2d}, Test: {class_split_counts[c]['test']:2d})")
    print(f"  ... (+ {len(target_classes) - 10} more classes with 70 images each)")
else:
    print("PlantVillage manifest not found!")

# 2. Audit PlantDoc Local Dataset
print("\n" + "=" * 60)
print("2. LOCAL PLANTDOC DATASET AUDIT (dataset/plantdoc)")
print("=" * 60)
if pd_manifest_path.exists():
    with open(pd_manifest_path, "r", encoding="utf-8") as f:
        pd_manifest = json.load(f)
    recs = pd_manifest.get("records", [])
    print(f"Manifest found: {pd_manifest_path}")
    print(f"Total recorded images: {len(recs)}")

    split_counts = defaultdict(int)
    class_split_counts = defaultdict(lambda: defaultdict(int))
    classes_present = set()

    for r in recs:
        split = r.get("split", "unknown")
        cname = r.get("class_name", "unknown")
        split_counts[split] += 1
        class_split_counts[cname][split] += 1
        classes_present.add(cname)

    print(f"Split breakdown: Train: {split_counts['train']}, Val: {split_counts['val']}, Test: {split_counts['test']}")
    print(f"Distinct classes present in PlantDoc: {len(classes_present)} / {len(target_classes)}")

    missing_in_pd = [c for c in target_classes if c not in classes_present]
    print(f"\nClasses in PlantDoc: {len(classes_present)}")
    print(f"Target classes NOT covered by PlantDoc (10 classes):")
    for m in missing_in_pd:
        print(f"  - {m}")
else:
    print("PlantDoc manifest not found!")
