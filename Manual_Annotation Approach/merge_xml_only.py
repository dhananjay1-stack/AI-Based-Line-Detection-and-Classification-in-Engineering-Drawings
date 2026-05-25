import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

def merge_xml_only():
    base_dir = Path(r'C:\Users\Admin\line_detection\Approach 2\Annotation\feature_visible')
    
    # Updated directory name where the XMLs are located
    img_dir = base_dir / 'job_3939604_annotations_cvat for images 1.1'
    target_xml = base_dir / 'annotations.xml'
    
    print("Merging annotations.xml...")
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
            print("Successfully saved merged annotations.xml")
            
        print("\nCleaning up old folder...")
        try:
            shutil.rmtree(img_dir)
            print(f"Deleted {img_dir.name}")
        except Exception as e:
            print(f"[!] Could not delete folder automatically. Error: {e}")
    else:
        print(f"Directory not found: {img_dir}")

if __name__ == '__main__':
    merge_xml_only()
