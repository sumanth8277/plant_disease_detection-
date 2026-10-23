import urllib.request
import json
import os

headers = {'User-Agent': 'Mozilla/5.0'}

def get_contents(path=''):
    url = f'https://api.github.com/repos/Gladiator07/Harvestify/contents/{path}'
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        print(f"Error fetching {path}: {e}")
        return []

def explore(path='', indent=0, max_depth=2):
    if indent > max_depth:
        return
    items = get_contents(path)
    if isinstance(items, dict) and 'message' in items:
        print(' ' * indent + f"API Message: {items['message']}")
        return
    if not isinstance(items, list):
        return
    for item in items:
        prefix = '  ' * indent
        item_type = item.get('type')
        name = item.get('name')
        size = item.get('size', 0)
        print(f"{prefix}[{item_type}] {name} ({size} bytes)")
        if item_type == 'dir':
            explore(item.get('path', ''), indent + 1, max_depth=max_depth)

if __name__ == '__main__':
    print("--- Harvestify Repository Structure ---")
    explore(max_depth=2)
