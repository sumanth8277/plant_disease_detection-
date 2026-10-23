import requests
import os

base_raw_url = "https://raw.githubusercontent.com/ravikant-diwakar/AgriSens/master/"
files_to_download = [
    ("README.md", "README.md"),
    ("DISEASE-GUIDE.md", "DISEASE-GUIDE.md"),
    ("Datasets/Fertilizer_recommendation.csv", "Fertilizer_recommendation.csv"),
    ("Datasets/README.md", "Datasets_README.md"),
    ("PLANT-DISEASE-IDENTIFICATION/main.py", "main.py"),
    ("PLANT-DISEASE-IDENTIFICATION/training_hist.json", "training_hist.json")
]

out_dir = os.path.join("reports", "agrisens_reference")
os.makedirs(out_dir, exist_ok=True)

headers = {'User-Agent': 'Mozilla/5.0'}

for remote_rel, local_name in files_to_download:
    url = base_raw_url + remote_rel
    dest = os.path.join(out_dir, local_name)
    print(f"Fetching {remote_rel}...", flush=True)
    try:
        r = requests.get(url, headers=headers, timeout=15)
        if r.status_code == 200:
            with open(dest, 'wb') as fp:
                fp.write(r.content)
            print(f"Saved {local_name} ({len(r.content)} bytes)", flush=True)
        else:
            print(f"Failed with status {r.status_code}", flush=True)
    except Exception as e:
        print(f"Error fetching {remote_rel}: {e}", flush=True)

print("AgriSens fetch completed!", flush=True)
