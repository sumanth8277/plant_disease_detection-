import requests
import re

s = requests.Session()
r_get = s.get('http://127.0.0.1:5000/upload')
csrf_match = re.search(r'name="csrf_token" value="([^"]+)"', r_get.text)

token = csrf_match.group(1) if csrf_match else ''
print('CSRF token found:', bool(token))

# Let's check a smaller image from PlantVillage (under 1MB)
test_file = r'dataset\raw\PlantVillage\Apple___Apple_scab\01a66316-0e98-4d3b-a56f-d78752cd043f___FREC_Scab 3003.JPG'
with open(test_file, 'rb') as f:
    files = {'leaf_image': ('apple_leaf.jpg', f, 'image/jpeg')}
    data = {'csrf_token': token}
    resp = s.post('http://127.0.0.1:5000/upload', files=files, data=data, allow_redirects=True)
    print('Final status code:', resp.status_code)
    print('Final URL:', resp.url)
    
    alerts = re.findall(r'<div class="alert[^>]*>(.*?)</div>', resp.text, re.DOTALL)
    for a in alerts:
        clean_a = re.sub(r'<[^>]+>', ' ', a).strip()
        print('Alert:', clean_a)

    if 'Targeted Commercial Remedy &amp; Fertilizer' in resp.text or 'Targeted Commercial Remedy & Fertilizer' in resp.text:
        print('\nSUCCESS: Targeted Commercial Remedy & Fertilizer card is rendered!')
        snippet = re.search(r'(Targeted Commercial Remedy.*?Verified Product Sourcing)', resp.text, re.DOTALL)
        if snippet:
            clean = re.sub(r'<[^>]+>', ' ', snippet.group(1))
            clean = ' '.join(clean.split())
            print('\nRendered Card Summary:\n', clean[:400])

