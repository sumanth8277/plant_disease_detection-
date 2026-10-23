import re
import requests

# 1. Test Home page
r_home = requests.get('http://127.0.0.1:5000/')
nav_match = re.search(r'<nav class="nav-links">(.*?)</nav>', r_home.text, re.DOTALL)
nav_html = nav_match.group(1) if nav_match else ''
print('--- Home Page Navbar ---')
print('Has Log In:', 'Log In' in nav_html)
print('Has Register:', 'Register' in nav_html)
print('Has Health in nav:', 'Health' in nav_html)

print('\n--- Home Page Feature Cards ---')
print('Has CNN Model card:', 'CNN Model' in r_home.text)
print('Has MobileNetV2 Deep Learning:', 'MobileNetV2 Deep Learning' in r_home.text)

# 2. Test upload and result page
s = requests.Session()
r_upload = s.get('http://127.0.0.1:5000/upload')
csrf = re.search(r'name="csrf_token" value="([^"]+)"', r_upload.text).group(1)

test_file = r'dataset\plantvillage_candidate\images\Apple___Apple_scab\01a66316-0e98-4d3b-a56f-d78752cd043f___FREC_Scab 3003.JPG'
with open(test_file, 'rb') as f:
    resp = s.post('http://127.0.0.1:5000/upload', files={'leaf_image': ('apple.jpg', f, 'image/jpeg')}, data={'csrf_token': csrf}, allow_redirects=True)

print('\n--- Result Page Verification ---')
print('Result page status:', resp.status_code)
print('Has Gemini Analysis card:', 'Gemini Supplementary Botanical Analysis' in resp.text)
print('Has Audit Log box:', 'Audit Log:' in resp.text)
print('Has Database offline box:', 'Database offline:' in resp.text)
print('Has Diagnosis & Treatment:', ('Apple' in resp.text and 'Scab' in resp.text))
print('Has Verified Indian Treatment:', 'Verified Indian Agricultural Treatment' in resp.text)
