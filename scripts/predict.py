"""
Steel Surface Defect Detection - Inference Script
Author: Mohammed Abdulqawi Alezzi Saleh
"""

import os
import argparse
from ultralytics import YOLO


def predict(args):
    """Run inference on images using trained model."""
    
    print("=" * 50)
    print("Steel Surface Defect Detection - Inference")
    print("=" * 50)
    
    # Load trained model
    print(f"\nLoading model: {args.weights}")
    model = YOLO(args.weights)
    
    # Run inference
    print(f"\nRunning inference on: {args.source}")
    results = model.predict(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        save=args.save,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
        project=args.project,
        name=args.name,
        device=args.device
    )
    
    # Print results
    print("\n" + "=" * 50)
    print("Detection Results:")
    print("=" * 50)
    
    class_names = ['crazing', 'inclusion', 'patches', 
                   'pitted_surface', 'rolled-in_scale', 'scratches']
    
    for i, result in enumerate(results):
        print(f"\nImage {i + 1}: {result.path}")
        boxes = result.boxes
        if len(boxes) > 0:
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                print(f"  - {class_names[cls_id]}: {conf:.2%} confidence")
        else:
            print("  - No defects detected")
    
    print("\n" + "=" * 50)
    print(f"Results saved to: {args.project}/{args.name}")
    print("=" * 50)
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Run inference for Steel Defect Detection')
    
    # Model arguments
    parser.add_argument('--weights', type=str, default='weights/best.pt',
                        help='Path to trained model weights')
    parser.add_argument('--source', type=str, required=True,
                        help='Image source (file, directory, URL)')
    
    # Inference arguments
    parser.add_argument('--conf', type=float, default=0.5,
                        help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.7,
                        help='IoU threshold for NMS')
    parser.add_argument('--device', type=str, default='0',
                        help='Device to use')
    
    # Output arguments
    parser.add_argument('--save', action='store_true', default=True,
                        help='Save results')
    parser.add_argument('--save-txt', action='store_true',
                        help='Save results as .txt')
    parser.add_argument('--save-conf', action='store_true',
                        help='Save confidence scores')
    parser.add_argument('--project', type=str, default='runs/predict',
                        help='Project directory')
    parser.add_argument('--name', type=str, default='results',
                        help='Experiment name')
    
    args = parser.parse_args()
    
    # Run inference
    predict(args)


if __name__ == '__main__':
    main()
