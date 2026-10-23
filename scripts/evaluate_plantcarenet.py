import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from services.image_preprocessing import load_and_preprocess_image

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger("eval_plantcarenet")

CANDIDATES_DIR = BASE_DIR / "models" / "candidates"
REPORTS_DIR = BASE_DIR / "reports" / "candidates"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

candidate_path = CANDIDATES_DIR / "plantcarenet_candidate.keras"
pv_manifest = BASE_DIR / "dataset" / "plantvillage_candidate" / "manifest.json"
pd_manifest = BASE_DIR / "dataset" / "plantdoc" / "manifest.json"

with open(pv_manifest, "r", encoding="utf-8") as f:
    pv_data = json.load(f)
with open(pd_manifest, "r", encoding="utf-8") as f:
    pd_data = json.load(f)
with open(BASE_DIR / "models" / "class_names.json", "r", encoding="utf-8") as f:
    class_names = json.load(f)

pv_test_recs = [r for r in pv_data["records"] if r["split"] == "test"]
pd_test_recs = [r for r in pd_data["records"] if r["split"] == "test"]
comb_recs = pv_test_recs + pd_test_recs

logger.info("Loading PlantCareNet model from %s...", candidate_path)
model = tf.keras.models.load_model(str(candidate_path))
target_size = (128, 128)

def evaluate_subset(recs, desc="Test Set"):
    logger.info("Evaluating %s (%d images)...", desc, len(recs))
    y_true, y_pred = [], []
    for r in recs:
        try:
            img = load_and_preprocess_image(r["filepath"], target_size=target_size)
            pred = model.predict(np.expand_dims(img, axis=0), verbose=0)
            y_true.append(int(r["class_index"]))
            y_pred.append(int(np.argmax(pred[0])))
        except Exception as e:
            logger.warning("Error processing %s: %s", r["filepath"], e)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    acc = float(np.mean(y_true == y_pred))
    
    unique_present = sorted(list(set(y_true) | set(y_pred)))
    rep = classification_report(
        y_true, 
        y_pred, 
        labels=unique_present,
        target_names=[class_names[i] for i in unique_present],
        output_dict=True, 
        zero_division=0
    )
    return acc, rep, y_true, y_pred

pv_acc, pv_rep, _, _ = evaluate_subset(pv_test_recs, "PlantVillage Lab Test")
pd_acc, pd_rep, _, _ = evaluate_subset(pd_test_recs, "PlantDoc Field Test")
comb_acc, comb_rep, comb_true, comb_pred = evaluate_subset(comb_recs, "Combined Test")

logger.info("=== PlantCareNet Candidate Benchmark Results ===")
logger.info(" - PlantVillage Lab Test Accuracy: %.2f%%", pv_acc * 100)
logger.info(" - PlantDoc Field Test Accuracy:    %.2f%%", pd_acc * 100)
logger.info(" - Overall Combined Test Accuracy:  %.2f%% (Macro F1: %.4f)", comb_acc * 100, comb_rep["macro avg"]["f1-score"])

# Plot confusion matrix
cm = confusion_matrix(comb_true, comb_pred, labels=list(range(len(class_names))))
fig, ax = plt.subplots(figsize=(14, 12))
cax = ax.matshow(cm, cmap="Blues")
fig.colorbar(cax)
ax.set_title("PlantCareNet Candidate Model - Confusion Matrix", fontsize=14, pad=20)
ax.set_xlabel("Predicted Class Index", fontsize=11)
ax.set_ylabel("True Class Index", fontsize=11)
plt.tight_layout()
cm_path = REPORTS_DIR / "plantcarenet_candidate_confusion_matrix.png"
plt.savefig(cm_path, dpi=200)
plt.close()

eval_summary = {
    "candidate_model": "plantcarenet_candidate.keras",
    "architecture": "Custom PlantCareNet CNN (Muhaimin008)",
    "input_shape": [128, 128, 3],
    "lab_test_accuracy_percent": round(pv_acc * 100, 2),
    "field_test_accuracy_percent": round(pd_acc * 100, 2),
    "combined_test_accuracy_percent": round(comb_acc * 100, 2),
    "combined_macro_f1": round(comb_rep["macro avg"]["f1-score"], 4),
    "plantvillage_samples_evaluated": len(pv_test_recs),
    "plantdoc_samples_evaluated": len(pd_test_recs),
    "total_test_samples": len(comb_recs)
}

eval_json_path = REPORTS_DIR / "plantcarenet_candidate_eval.json"
with open(eval_json_path, "w", encoding="utf-8") as f:
    json.dump(eval_summary, f, indent=2)

import hashlib
def compute_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as fp:
        while chunk := fp.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

metadata = {
    "model_version": "PlantCareNet-Candidate-20261023",
    "architecture": "Custom PlantCareNet CNN (Conv2D + BatchNorm + Dropout)",
    "source_repo": "Muhaimin008/Plant-disease-classification-and-recommendations",
    "input_shape": [128, 128, 3],
    "num_classes": len(class_names),
    "sha256": compute_sha256(candidate_path),
    "file_size_bytes": candidate_path.stat().st_size,
    "metrics": eval_summary
}
with open(CANDIDATES_DIR / "plantcarenet_candidate_metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

logger.info("Evaluation complete! Summary saved to %s and metadata saved to %s", eval_json_path, CANDIDATES_DIR / "plantcarenet_candidate_metadata.json")
