import requests

# Health
r = requests.get('http://127.0.0.1:5000/api/health')
print("Health:", r.json())

# Classes
r2 = requests.get('http://127.0.0.1:5000/api/classes')
d = r2.json()
print(f"\nClasses: {len(d['classes'])}")
for c in d['classes']:
    print(f"  {c['index']:2d}: {c['name']:20s} {c['color']}")

# Checkpoints
r3 = requests.get('http://127.0.0.1:5000/api/checkpoints')
d3 = r3.json()
print(f"\nCheckpoints: {len(d3['checkpoints'])}")
for c in d3['checkpoints'][:5]:
    print(f"  {c['name']:30s} {c['size_mb']} MB")
if len(d3['checkpoints']) > 5:
    print(f"  ... and {len(d3['checkpoints'])-5} more")

print("\nAll API endpoints OK!")
