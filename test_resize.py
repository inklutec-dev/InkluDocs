from PIL import Image
import os, base64, httpx, json

img_path = '/app/data/results/1/4/p1_img1.jpeg'
img = Image.open(img_path)
orig_size = os.path.getsize(img_path)
print(f"Original: {img.size}, {orig_size} bytes")

# Resize
img.thumbnail((800, 800), Image.LANCZOS)
small_path = '/tmp/test_small.jpg'
img.save(small_path, 'JPEG', quality=85)
small_size = os.path.getsize(small_path)
print(f"Resized: {img.size}, {small_size} bytes")

# Test with resized image
with open(small_path, 'rb') as f:
    img_b64 = base64.b64encode(f.read()).decode()
print(f"Base64: {len(img_b64)} chars")

OLLAMA_URL = 'http://172.20.0.1:11434'
print("Sending to Ollama...")
r = httpx.post(OLLAMA_URL + '/api/generate', json={
    'model': 'qwen3-vl:8b',
    'prompt': 'Beschreibe dieses Bild in 2 Saetzen auf Deutsch.',
    'images': [img_b64],
    'stream': False,
    'options': {'temperature': 0.3}
}, timeout=600)
data = r.json()
resp = data.get('response', '')
print(f"Response ({len(resp)} chars):")
print(resp[:800])
