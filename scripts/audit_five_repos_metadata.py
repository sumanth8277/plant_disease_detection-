import urllib.request
import json

repos = [
    "spMohanty/PlantVillage-Dataset",
    "pratikkayal/PlantDoc-Dataset",
    "pratikkayal/PlantDoc-Object-Detection-Dataset",
    "MuhammadEhsan02/AI-Powered-Plant-Disease-Detection-and-Fertilizer-Recommendation-System",
    "Gladiator07/Harvestify"
]

headers = {'User-Agent': 'Mozilla/5.0'}
for r in repos:
    url = f"https://api.github.com/repos/{r}"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode())
            lic = data.get("license")
            lic_name = lic.get("name") if lic else "None specified in GitHub metadata"
            print(f"=== {r} ===")
            print(f"  Size: {data.get('size')} KB")
            print(f"  Default Branch: {data.get('default_branch')}")
            print(f"  License: {lic_name}")
            print(f"  Description: {data.get('description')}")
            print()
    except Exception as e:
        print(f"Error fetching {r}: {e}")
