import urllib.request
import json
import os

headers = {'User-Agent': 'Mozilla/5.0'}

def get_contents(path=''):
    url = f'https://api.github.com/repos/MuhammadEhsan02/AI-Powered-Plant-Disease-Detection-and-Fertilizer-Recommendation-System/contents/{path}'
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        print(f"Error fetching {path}: {e}")
        return []

def explore(path='', indent=0):
    items = get_contents(path)
    if isinstance(items, dict) and 'message' in items:
        print(' ' * indent + f"API Message: {items['message']}")
        return
    for item in items:
        prefix = '  ' * indent
        item_type = item.get('type')
        name = item.get('name')
        size = item.get('size', 0)
        print(f"{prefix}[{item_type}] {name} ({size} bytes)")
        if item_type == 'dir':
            explore(item.get('path', ''), indent + 1)

if __name__ == '__main__':
    print("--- Repository Tree ---")
    explore()
