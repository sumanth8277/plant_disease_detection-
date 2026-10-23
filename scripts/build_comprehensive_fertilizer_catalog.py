import json
import csv
import os

with open('models/class_names.json', 'r') as f:
    class_names = json.load(f)

with open('reports/candidates/ehsan_disease_fertilizer_catalog.json', 'r', encoding='utf-8') as f:
    ehsan_items = json.load(f)

ehsan_map = {}
for item in ehsan_items:
    cid = item.get('class_id')
    if cid:
        ehsan_map[cid] = item

# Also load existing plantdoc agronomic guide if available
plantdoc_guide = {}
pd_path = 'reports/candidates/plantdoc_agronomic_recovery_guide.json'
if os.path.exists(pd_path):
    with open(pd_path, 'r', encoding='utf-8') as f:
        plantdoc_guide = json.load(f)

unified_catalog = []

for i, cname in enumerate(class_names):
    crop = cname.split('___')[0].replace('_', ' ')
    disease = cname.split('___')[1].replace('_', ' ')
    is_healthy = 'healthy' in cname.lower()
    
    ehsan_info = ehsan_map.get(cname, {})
    if not ehsan_info:
        # Fallback check for normalized name
        norm = cname.replace('___', ': ').replace('_', ' ')
        for k, v in ehsan_map.items():
            if cname.split('___')[1].lower() in k.lower():
                ehsan_info = v
                break

    pd_info = plantdoc_guide.get(cname, {})

    entry = {
        "class_id": i,
        "class_name": cname,
        "crop": crop,
        "condition": disease,
        "is_healthy": is_healthy,
        "description": ehsan_info.get("description", pd_info.get("symptoms", f"{crop} affected by {disease}.")),
        "possible_steps": ehsan_info.get("possible_steps", pd_info.get("cultural_controls", "Ensure balanced soil fertilization and inspect regularly.")),
        "recommended_supplement_name": ehsan_info.get("supplement_name", "Balanced N-P-K Organic Fertilizer" if is_healthy else "Copper-based Fungicide/Bactericide"),
        "supplement_category": "Growth Promoter / Fertilizer" if is_healthy else "Fungicide / Bactericide / Pesticide",
        "buy_or_source_link": ehsan_info.get("buy_link", "https://kissanghar.pk/ or local certified agricultural dealer"),
        "product_image_url": ehsan_info.get("supplement_image", ""),
        "disease_reference_image": ehsan_info.get("image_url", ""),
        "active_ingredients_and_guidance": pd_info.get("chemical_controls", "Follow manufacturer label dosage strictly. Apply during early morning or late evening.")
    }
    unified_catalog.append(entry)

out_file = 'reports/candidates/comprehensive_crop_recovery_and_fertilizer_catalog.json'
with open(out_file, 'w', encoding='utf-8') as f:
    json.dump(unified_catalog, f, indent=2)

print(f"Unified catalog compiled with {len(unified_catalog)} classes into {out_file}!")
