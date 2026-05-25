import json
from pathlib import Path

# 1. Check classes_cleaned.json
with open(r'C:\Users\Admin\line_detection\Approach 2\pipeline_output\classes_cleaned.json') as f:
    cmap = json.load(f)
print('Classes in cleaned mapping:')
for k, v in cmap.items():
    print(f'  {v}: {k}')
has_section = 'Section' in str(cmap)
print(f'Section_hatching in class map? {has_section}')

# 2. Check class_distribution.csv  
with open(r'C:\Users\Admin\line_detection\Approach 2\pipeline_output\class_distribution.csv') as f:
    lines = f.readlines()
print('\nClass distribution (REMOVED entries):')
for l in lines:
    if 'REMOVED' in l:
        print(f'  {l.strip()}')

# 3. Check patch class distribution
with open(r'C:\Users\Admin\line_detection\Approach 2\dataset_patches\metadata\class_distribution_patches.csv') as f:
    patch_lines = f.readlines()
print('\nPatch class distribution:')
for l in patch_lines:
    print(f'  {l.strip()}')
has_in_patches = any('Section' in l for l in patch_lines)
print(f'Section_hatching in patches? {has_in_patches}')

# 4. Check pipeline constants
with open(r'C:\Users\Admin\line_detection\Approach 2\dataset_pipeline.py') as f:
    code = f.read()
import re
m = re.search(r'CLASSES_TO_REMOVE = \{(.+?)\}', code)
print(f'\nIn dataset_pipeline.py:')
print(f'  {m.group(0)}')
