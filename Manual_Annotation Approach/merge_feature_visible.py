import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

def merge_feature_visible():
    base_dir = Path(r'C:\Users\Admin\line_detection\Approach 2\Annotation\feature_visible')
    seg_dir = base_dir / 'segmentation'
    img_dir = base_dir / 'image_1.1'
    
    # Target paths
    target_seg_mask_dir = base_dir / 'job_merged_segmentation mask 1.1'
    target_seg_class = target_seg_mask_dir / 'SegmentationClass'
    target_seg_obj = target_seg_mask_dir / 'SegmentationObject'
    target_img_sets = target_seg_mask_dir / 'ImageSets' / 'Segmentation'
    target_xml = base_dir / 'annotations.xml'
    
    print("1. Creating target directories...")
    target_seg_class.mkdir(parents=True, exist_ok=True)
    target_seg_obj.mkdir(parents=True, exist_ok=True)
    target_img_sets.mkdir(parents=True, exist_ok=True)
    
    frames_found = set()
    
    print("\n2. Merging Segmentation Masks...")
    if seg_dir.exists():
        for root, _, files in os.walk(seg_dir):
            if 'SegmentationClass' in root:
                for f in files:
                    if f.endswith('.png'):
                        shutil.copy2(Path(root) / f, target_seg_class / f)
                        frames_found.add(f.replace('.png', ''))
                        print(f"  Copied mask: {f}")
            elif 'SegmentationObject' in root:
                for f in files:
                    if f.endswith('.png'):
                        shutil.copy2(Path(root) / f, target_seg_obj / f)
    
    print("\n3. Creating labelmap.txt and ImageSets...")
    with open(target_seg_mask_dir / 'labelmap.txt', 'w') as f:
        f.write('# label:color_rgb:parts:actions\n')
        f.write('Feature_Visible:184,61,245::\n')
        f.write('background:0,0,0::\n')
        
    with open(target_img_sets / 'default.txt', 'w') as f:
        for frame in sorted(frames_found):
            f.write(f"{frame}\n")
            
    print("\n4. Merging annotations.xml...")
    base_tree = None
    base_root = None
    
    if img_dir.exists():
        for root, _, files in os.walk(img_dir):
            for f in files:
                if f.endswith('.xml'):
                    xml_path = Path(root) / f
                    try:
                        tree = ET.parse(xml_path)
                        root_elem = tree.getroot()
                        
                        if base_tree is None:
                            # Keep the first XML as our base structure
                            base_tree = tree
                            base_root = root_elem
                            print(f"  Base XML: {xml_path.parent.name}/{f}")
                        else:
                            # For subsequent XMLs, find all <image> tags and append to base
                            images_added = 0
                            for img_elem in root_elem.findall('image'):
                                base_root.append(img_elem)
                                images_added += 1
                            print(f"  Merged {images_added} images from {xml_path.parent.name}/{f}")
                    except Exception as e:
                        print(f"  Error parsing {xml_path}: {e}")
                        
    if base_tree is not None:
        # Save the merged XML to the root of feature_visible
        base_tree.write(target_xml, encoding='utf-8', xml_declaration=True)
        print("  Successfully saved merged annotations.xml")
        
    print("\n5. Cleaning up old folders...")
    try:
        if seg_dir.exists():
            shutil.rmtree(seg_dir)
            print("  Deleted old 'segmentation' folder.")
        if img_dir.exists():
            shutil.rmtree(img_dir)
            print("  Deleted old 'image_1.1' folder.")
    except Exception as e:
        print(f"\n  [!] Could not delete old folders automatically.")
        print(f"  [!] Error: {e}")
        print("  [!] Please close any File Explorer windows or image viewers that have these folders open, and delete them manually.")

    print("\nDone! Exact structure successfully created.")

if __name__ == '__main__':
    merge_feature_visible()
