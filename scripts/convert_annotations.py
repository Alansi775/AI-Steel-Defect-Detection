"""
Steel Surface Defect Detection - Annotation Converter
Converts Pascal VOC (XML) annotations to YOLO format
Author: Mohammed Abdulqawi Alezzi Saleh
"""

import os
import xml.etree.ElementTree as ET
import argparse
from tqdm import tqdm


# Class names for NEU-DET dataset
CLASSES = ['crazing', 'inclusion', 'patches', 
           'pitted_surface', 'rolled-in_scale', 'scratches']


def convert_bbox_to_yolo(size, box):
    """
    Convert bounding box from VOC format to YOLO format.
    
    Args:
        size: (width, height) of the image
        box: (xmin, ymin, xmax, ymax) bounding box
    
    Returns:
        (x_center, y_center, width, height) normalized to [0, 1]
    """
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    
    x_center = (box[0] + box[2]) / 2.0
    y_center = (box[1] + box[3]) / 2.0
    w = box[2] - box[0]
    h = box[3] - box[1]
    
    return (x_center * dw, y_center * dh, w * dw, h * dh)


def convert_annotation(xml_path, output_path):
    """
    Convert a single XML annotation file to YOLO format.
    
    Args:
        xml_path: Path to input XML file
        output_path: Path to output TXT file
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Get image size
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    
    # Convert each object
    with open(output_path, 'w') as out_file:
        for obj in root.iter('object'):
            # Get class name
            cls_name = obj.find('name').text.lower().replace(' ', '_')
            
            if cls_name not in CLASSES:
                print(f"Warning: Unknown class '{cls_name}' in {xml_path}")
                continue
            
            cls_id = CLASSES.index(cls_name)
            
            # Get bounding box
            xmlbox = obj.find('bndbox')
            box = (
                float(xmlbox.find('xmin').text),
                float(xmlbox.find('ymin').text),
                float(xmlbox.find('xmax').text),
                float(xmlbox.find('ymax').text)
            )
            
            # Convert to YOLO format
            yolo_bbox = convert_bbox_to_yolo((w, h), box)
            
            # Write to file
            out_file.write(f"{cls_id} {' '.join([f'{x:.6f}' for x in yolo_bbox])}\n")


def process_dataset(input_dir, output_dir):
    """
    Process all XML files in a directory.
    
    Args:
        input_dir: Directory containing XML files
        output_dir: Directory to save YOLO format TXT files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    xml_files = [f for f in os.listdir(input_dir) if f.endswith('.xml')]
    
    print(f"Converting {len(xml_files)} annotations...")
    
    for xml_file in tqdm(xml_files):
        xml_path = os.path.join(input_dir, xml_file)
        txt_name = xml_file.replace('.xml', '.txt')
        txt_path = os.path.join(output_dir, txt_name)
        
        convert_annotation(xml_path, txt_path)
    
    print(f"Done! Saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Convert VOC annotations to YOLO format')
    
    parser.add_argument('--input', type=str, required=True,
                        help='Input directory containing XML files')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory for YOLO format files')
    
    args = parser.parse_args()
    
    process_dataset(args.input, args.output)


if __name__ == '__main__':
    main()
