import requests
import os

base_raw_url = "https://raw.githubusercontent.com/Gladiator07/Harvestify/master/"
files_to_download = [
    ("README.md", "README.md"),
    ("app/utils/disease.py", "disease.py"),
    ("app/utils/fertilizer.py", "fertilizer.py"),
    ("app/utils/model.py", "model.py"),
    ("Data-processed/fertilizer.csv", "fertilizer.csv"),
    ("app/app.py", "app.py")
]

out_dir = os.path.join("reports", "harvestify_reference")
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

print("Fetch completed!", flush=True)
