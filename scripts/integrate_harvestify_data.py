import json
import csv
import os

catalog_path = "model/comprehensive_crop_recovery_and_fertilizer_catalog.json"
with open(catalog_path, "r", encoding="utf-8") as f:
    catalog = json.load(f)

# Load Harvestify fertilizer baseline
h_fert_csv = "reports/harvestify_reference/fertilizer.csv"
h_fert_data = {}
if os.path.exists(h_fert_csv):
    with open(h_fert_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            crop = r.get("Crop", "").strip().lower()
            if crop:
                h_fert_data[crop] = {
                    "N": r.get("N"),
                    "P": r.get("P"),
                    "K": r.get("K"),
                    "pH": r.get("pH"),
                    "soil_moisture": r.get("soil_moisture")
                }

# Standard NPK baselines for other crops if missing
crop_baselines = {
    "apple": {"N": 20, "P": 125, "K": 200, "pH": 6.5},
    "blueberry": {"N": 40, "P": 30, "K": 40, "pH": 4.8},
    "cherry (including sour)": {"N": 50, "P": 30, "K": 60, "pH": 6.5},
    "corn (maize)": {"N": 80, "P": 40, "K": 20, "pH": 5.5},
    "grape": {"N": 20, "P": 125, "K": 200, "pH": 6.0},
    "orange": {"N": 20, "P": 10, "K": 10, "pH": 6.0},
    "peach": {"N": 60, "P": 40, "K": 80, "pH": 6.5},
    "pepper, bell": {"N": 80, "P": 40, "K": 60, "pH": 6.2},
    "potato": {"N": 100, "P": 60, "K": 80, "pH": 5.8},
    "raspberry": {"N": 45, "P": 35, "K": 50, "pH": 6.0},
    "soybean": {"N": 20, "P": 60, "K": 40, "pH": 6.5},
    "squash": {"N": 70, "P": 50, "K": 70, "pH": 6.5},
    "strawberry": {"N": 60, "P": 40, "K": 80, "pH": 6.0},
    "tomato": {"N": 100, "P": 50, "K": 60, "pH": 6.5}
}

for item in catalog:
    crop_name = item.get("crop", "").strip().lower()
    base = crop_baselines.get(crop_name)
    if not base:
        for k, v in crop_baselines.items():
            if k in crop_name or crop_name in k:
                base = v
                break

    if base:
        item["optimal_npk_baseline"] = {
            "Nitrogen_ratio": base["N"],
            "Phosphorus_ratio": base["P"],
            "Potassium_ratio": base["K"],
            "optimal_soil_pH": base["pH"]
        }
        item["npk_corrective_guidance"] = {
            "if_nitrogen_low": "Amend with composted farmyard manure, blood meal, or high-N urea (46-0-0).",
            "if_phosphorus_low": "Apply steamed bone meal or rock phosphate, maintaining soil pH near 6.5 for optimal uptake.",
            "if_potassium_low": "Incorporate sulphate of potash (0-0-50) or organic wood ash to enhance natural disease resistance."
        }

with open(catalog_path, "w", encoding="utf-8") as f:
    json.dump(catalog, f, indent=2)

print(f"Successfully integrated Harvestify N-P-K soil nutrient baselines into {catalog_path}!")
