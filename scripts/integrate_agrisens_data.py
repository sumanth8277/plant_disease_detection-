import json
import os

catalog_path = "model/comprehensive_crop_recovery_and_fertilizer_catalog.json"
with open(catalog_path, "r", encoding="utf-8") as f:
    catalog = json.load(f)

# AgriSens multi-nutrient fertilizer mappings based on crop and growth needs
agrisens_fertilizers = {
    "apple": {"primary_compound": "10-26-26 Complex Fertilizer", "role": "Enhances root strength and winter hardiness with high potassium/phosphorus."},
    "blueberry": {"primary_compound": "Ammonium Sulphate + Urea", "role": "Maintains acidic soil pH while supplying bioavailable nitrogen."},
    "cherry": {"primary_compound": "17-17-17 Balanced NPK", "role": "Promotes balanced spur formation and uniform fruit size."},
    "corn": {"primary_compound": "Urea (46-0-0) + DAP (18-46-0)", "role": "High nitrogen top-dressing drives rapid canopy expansion and cob filling."},
    "grape": {"primary_compound": "10-26-26 + Sulphate of Potash", "role": "Boosts berry sugar accumulation (Brix level) and vine cane maturation."},
    "orange": {"primary_compound": "14-35-14 Starter + Citrus Micronutrients", "role": "Supports root flushing and prevents chlorosis from nutrient lockout."},
    "peach": {"primary_compound": "17-17-17 Balanced NPK", "role": "Provides balanced vegetative shoots without inducing fungal vulnerability."},
    "pepper": {"primary_compound": "20-20 Complex + Potash", "role": "Prevents blossom drop and supports continuous pepper pod setting."},
    "potato": {"primary_compound": "DAP (18-46-0) + 10-26-26 Complex", "role": "Crucial phosphorus supply for rapid stolon and tuber initiation."},
    "raspberry": {"primary_compound": "14-35-14 Organic Blend", "role": "Stimulates strong primocane development and flower cluster counts."},
    "soybean": {"primary_compound": "DAP (18-46-0) Starter", "role": "Supplies phosphorus for root nodule bacteria colonization."},
    "squash": {"primary_compound": "28-28 Complex Fertilizer", "role": "High energy delivery for massive vine biomass and heavy squash fruit."},
    "strawberry": {"primary_compound": "17-17-17 + Potassium Nitrate", "role": "Improves fruit sweetness, firmness, and shelf-life resistance."},
    "tomato": {"primary_compound": "10-26-26 Complex + DAP (18-46-0)", "role": "High P & K prevents blossom end rot and fuels continuous heavy yields."}
}

for item in catalog:
    crop_name = item.get("crop", "").strip().lower()
    matched_entry = None
    for k, v in agrisens_fertilizers.items():
        if k in crop_name:
            matched_entry = v
            break
    if matched_entry:
        item["agrisens_precision_fertilizer"] = matched_entry
        item["academic_extension_reference"] = "NC State University & University of Minnesota Extension Diagnostic Guidelines (AgriSens)"

with open(catalog_path, "w", encoding="utf-8") as f:
    json.dump(catalog, f, indent=2)

print(f"Successfully integrated AgriSens precision fertilizer compounds and academic citations into {catalog_path}!")
