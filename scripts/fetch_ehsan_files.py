import urllib.request
import os

base_raw_url = "https://raw.githubusercontent.com/MuhammadEhsan02/AI-Powered-Plant-Disease-Detection-and-Fertilizer-Recommendation-System/main/"
files_to_download = [
    "README.md",
    "disease_info.csv",
    "supplement_info.csv",
    "CNN.py",
    "app.py"
]

out_dir = os.path.join("reports", "ehsan_reference")
os.makedirs(out_dir, exist_ok=True)

headers = {'User-Agent': 'Mozilla/5.0'}

for f in files_to_download:
    url = base_raw_url + f
    dest = os.path.join(out_dir, f)
    print(f"Downloading {f} -> {dest}...")
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as resp:
        content = resp.read()
        with open(dest, 'wb') as fp:
            fp.write(content)
    print(f"Saved {f} ({len(content)} bytes)")
