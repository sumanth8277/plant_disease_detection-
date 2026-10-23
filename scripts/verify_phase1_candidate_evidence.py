"""
Phase 1 Verification: Independent Evaluation of Unified Candidate vs. Production Model
======================================================================================
Verifies model weights, hashes, data leakage, and executes direct inference on
independent laboratory (PlantVillage) and field (PlantDoc) test sets.
"""

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from services.image_preprocessing import load_and_preprocess_image

def get_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def check_leakage(manifest_path: Path, name: str) -> Dict[str, Any]:
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    records = data.get("records", [])
    
    splits = {"train": set(), "val": set(), "test": set()}
    for r in records:
        s = r.get("split")
        p = r.get("filepath")
        if s in splits and p:
            splits[s].add(p)
            
    train_val_overlap = splits["train"].intersection(splits["val"])
    train_test_overlap = splits["train"].intersection(splits["test"])
    val_test_overlap = splits["val"].intersection(splits["test"])
    
    has_leakage = bool(train_val_overlap or train_test_overlap or val_test_overlap)
    
    return {
        "dataset_name": name,
        "total_records": len(records),
        "train_count": len(splits["train"]),
        "val_count": len(splits["val"]),
        "test_count": len(splits["test"]),
        "train_val_overlap": len(train_val_overlap),
        "train_test_overlap": len(train_test_overlap),
        "val_test_overlap": len(val_test_overlap),
        "has_data_leakage": has_leakage,
    }

def main():
    print("=====================================================================")
    print("PHASE 1: INDEPENDENT EMPIRICAL VERIFICATION OF CANDIDATE & PRODUCTION")
    print("=====================================================================")

    pv_manifest = BASE_DIR / "dataset" / "plantvillage_candidate" / "manifest.json"
    pd_manifest = BASE_DIR / "dataset" / "plantdoc" / "manifest.json"
    class_names_path = BASE_DIR / "models" / "class_names.json"
    
    cand_model_path = BASE_DIR / "models" / "candidates" / "unified_multidataset_model.keras"
    prod_model_path = BASE_DIR / "models" / "best_model.keras"
    
    # 1. Verify existence & compute hashes
    print("\n--- 1. Model Existence and SHA-256 Hashes ---")
    if not cand_model_path.exists():
        print(f"ERROR: Candidate model missing at {cand_model_path}")
        return
    if not prod_model_path.exists():
        print(f"ERROR: Production model missing at {prod_model_path}")
        return
        
    cand_sha = get_file_sha256(cand_model_path)
    prod_sha = get_file_sha256(prod_model_path)
    print(f"Candidate Path:  {cand_model_path}")
    print(f"Candidate SHA256: {cand_sha}")
    print(f"Production Path: {prod_model_path}")
    print(f"Production SHA256: {prod_sha}")
    
    # 2. Check Data Leakage
    print("\n--- 2. Dataset Separation & Data Leakage Audit ---")
    pv_leakage = check_leakage(pv_manifest, "PlantVillage Candidate")
    pd_leakage = check_leakage(pd_manifest, "PlantDoc Field Dataset")
    print(f"PlantVillage: {pv_leakage['total_records']} images (Train: {pv_leakage['train_count']}, Val: {pv_leakage['val_count']}, Test: {pv_leakage['test_count']})")
    print(f"  Leakage: train/test overlap = {pv_leakage['train_test_overlap']} (Pass: {not pv_leakage['has_data_leakage']})")
    print(f"PlantDoc:     {pd_leakage['total_records']} images (Train: {pd_leakage['train_count']}, Val: {pd_leakage['val_count']}, Test: {pd_leakage['test_count']})")
    print(f"  Leakage: train/test overlap = {pd_leakage['train_test_overlap']} (Pass: {not pd_leakage['has_data_leakage']})")

    # 3. Load Class Names
    with open(class_names_path, "r", encoding="utf-8") as f:
        class_names = json.load(f)
    num_classes = len(class_names)
    print(f"\nTotal Target Classes: {num_classes}")

    # 4. Load Models via TensorFlow
    print("\n--- 3. Loading Models into TensorFlow ---")
    import tensorflow as tf
    t0 = time.time()
    cand_model = tf.keras.models.load_model(str(cand_model_path), compile=False)
    prod_model = tf.keras.models.load_model(str(prod_model_path), compile=False)
    print(f"Models loaded in {time.time() - t0:.2f}s")
    print(f"Candidate output shape: {cand_model.output_shape} (Matches 38 classes: {cand_model.output_shape[-1] == num_classes})")
    print(f"Production output shape: {prod_model.output_shape} (Matches 38 classes: {prod_model.output_shape[-1] == num_classes})")

    # 5. Load Independent Test Sets
    with open(pv_manifest, "r", encoding="utf-8") as f:
        pv_records = [r for r in json.load(f)["records"] if r["split"] == "test"]
    with open(pd_manifest, "r", encoding="utf-8") as f:
        pd_records = [r for r in json.load(f)["records"] if r["split"] == "test"]
        
    print(f"\nPlantVillage Test Images: {len(pv_records)}")
    print(f"PlantDoc Field Test Images: {len(pd_records)}")
    print(f"Combined Independent Test Images: {len(pv_records) + len(pd_records)}")

    def evaluate_test_set(records: List[Dict[str, Any]], test_set_name: str):
        print(f"\nEvaluating: {test_set_name} ({len(records)} samples)...")
        imgs, y_true = [], []
        missing = 0
        for r in records:
            p = r["filepath"]
            if not os.path.exists(p):
                missing += 1
                continue
            try:
                img = load_and_preprocess_image(p, target_size=(224, 224))
                imgs.append(img)
                y_true.append(r["class_index"])
            except Exception as e:
                print(f"Failed to read {p}: {e}")
                missing += 1
                
        x = np.array(imgs, dtype=np.float32)
        y = np.array(y_true, dtype=np.int32)
        
        # Candidate predictions
        cand_probs = cand_model.predict(x, batch_size=32, verbose=0)
        cand_preds = np.argmax(cand_probs, axis=1)
        cand_acc = float(np.mean(cand_preds == y))
        
        # Production predictions
        prod_probs = prod_model.predict(x, batch_size=32, verbose=0)
        prod_preds = np.argmax(prod_probs, axis=1)
        prod_acc = float(np.mean(prod_preds == y))
        
        # Classification report for candidate
        unique_labels = sorted(list(set(y).union(set(cand_preds))))
        target_labels = [class_names[i] for i in unique_labels]
        cand_cr = classification_report(
            y, cand_preds, labels=unique_labels, target_names=target_labels, output_dict=True, zero_division=0
        )
        
        prod_unique_labels = sorted(list(set(y).union(set(prod_preds))))
        prod_target_labels = [class_names[i] for i in prod_unique_labels]
        prod_cr = classification_report(
            y, prod_preds, labels=prod_unique_labels, target_names=prod_target_labels, output_dict=True, zero_division=0
        )
        
        print(f"  --> Candidate Accuracy:  {cand_acc * 100:.2f}% (Macro F1: {cand_cr['macro avg']['f1-score']:.4f})")
        print(f"  --> Production Accuracy: {prod_acc * 100:.2f}% (Macro F1: {prod_cr['macro avg']['f1-score']:.4f})")
        
        return {
            "test_set_name": test_set_name,
            "sample_count": len(y),
            "candidate": {
                "accuracy_percent": round(cand_acc * 100, 2),
                "macro_precision": round(cand_cr["macro avg"]["precision"], 4),
                "macro_recall": round(cand_cr["macro avg"]["recall"], 4),
                "macro_f1": round(cand_cr["macro avg"]["f1-score"], 4),
                "weighted_f1": round(cand_cr["weighted avg"]["f1-score"], 4),
            },
            "production": {
                "accuracy_percent": round(prod_acc * 100, 2),
                "macro_precision": round(prod_cr["macro avg"]["precision"], 4),
                "macro_recall": round(prod_cr["macro avg"]["recall"], 4),
                "macro_f1": round(prod_cr["macro avg"]["f1-score"], 4),
                "weighted_f1": round(prod_cr["weighted avg"]["f1-score"], 4),
            },
            "candidate_per_class": {
                k: v for k, v in cand_cr.items() if k not in ("accuracy", "macro avg", "weighted avg")
            }
        }

    pv_eval = evaluate_test_set(pv_records, "PlantVillage (Laboratory Controlled)")
    pd_eval = evaluate_test_set(pd_records, "PlantDoc (Natural Field Conditions)")
    combined_eval = evaluate_test_set(pv_records + pd_records, "Combined Multi-Domain Test Set")

    claimed_accuracy = 72.85
    actual_accuracy = combined_eval["candidate"]["accuracy_percent"]
    acc_diff = abs(actual_accuracy - claimed_accuracy)
    is_verified = acc_diff < 0.1

    print("\n=====================================================================")
    print("EMPIRICAL VERIFICATION VERDICT:")
    print(f"  Claimed Accuracy: {claimed_accuracy}%")
    print(f"  Actual Measured Accuracy: {actual_accuracy}%")
    print(f"  Difference: {acc_diff:.4f}%")
    if is_verified:
        print("  STATUS: [VERIFIED]")
    else:
        print("  STATUS: [NOT VERIFIED - Discrepancy detected]")
    print("=====================================================================")

    out_dir = BASE_DIR / "reports" / "candidate_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "phase1_verified_evidence.json"
    
    report = {
        "verified_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "candidate_model_path": str(cand_model_path),
        "candidate_sha256": cand_sha,
        "production_model_path": str(prod_model_path),
        "production_sha256": prod_sha,
        "claimed_accuracy": claimed_accuracy,
        "actual_accuracy": actual_accuracy,
        "verification_verdict": "VERIFIED" if is_verified else "NOT VERIFIED",
        "leakage_audit": {
            "plantvillage": pv_leakage,
            "plantdoc": pd_leakage,
        },
        "evaluation_results": {
            "plantvillage_lab": pv_eval,
            "plantdoc_field": pd_eval,
            "combined": combined_eval,
        }
    }
    
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved empirical verification evidence to: {out_path}")

if __name__ == "__main__":
    main()
