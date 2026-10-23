import csv
import json

with open('reports/ehsan_reference/disease_info.csv', 'r', encoding='latin-1') as f:
    d_rows = list(csv.DictReader(f))

with open('reports/ehsan_reference/supplement_info.csv', 'r', encoding='latin-1') as f:
    s_rows = list(csv.DictReader(f))

print(f"Total disease_info rows: {len(d_rows)}")
print(f"Total supplement_info rows: {len(s_rows)}")

combined = []
for i in range(max(len(d_rows), len(s_rows))):
    d = d_rows[i] if i < len(d_rows) else {}
    s = s_rows[i] if i < len(s_rows) else {}
    item = {
        "index": i,
        "disease_label": d.get("disease_name", "").strip(),
        "class_id": s.get("disease_name", "").strip(),
        "description": d.get("description", "").strip(),
        "possible_steps": d.get("Possible Steps", "").strip(),
        "image_url": d.get("image_url", "").strip(),
        "supplement_name": s.get("supplement name", "").strip(),
        "supplement_image": s.get("supplement image", "").strip(),
        "buy_link": s.get("buy link", "").strip()
    }
    combined.append(item)
    print(f"[{i:02d}] {item['class_id']:40s} -> {item['supplement_name'][:50]}")

with open('reports/candidates/ehsan_disease_fertilizer_catalog.json', 'w', encoding='utf-8') as out_f:
    json.dump(combined, out_f, indent=2)

print("\nSaved catalog to reports/candidates/ehsan_disease_fertilizer_catalog.json")
