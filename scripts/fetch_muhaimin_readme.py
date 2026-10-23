import requests
import os

out_dir = os.path.join("reports", "muhaimin_reference")
os.makedirs(out_dir, exist_ok=True)

url = "https://raw.githubusercontent.com/Muhaimin008/Plant-disease-classification-and-recommendations/main/README.md"
r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=15)
if r.status_code == 200:
    with open(os.path.join(out_dir, "README.md"), "wb") as f:
        f.write(r.content)
    print("Downloaded README.md successfully!")
else:
    print("Failed to download README.md:", r.status_code)
