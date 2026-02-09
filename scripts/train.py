"""
Steel Surface Defect Detection - Training Script
Author: Mohammed Abdulqawi Alezzi Saleh
"""

import os
import argparse
from ultralytics import YOLO


def train(args):
    """Train YOLOv8 model for steel defect detection."""
    
    print("=" * 50)
    print("Steel Surface Defect Detection - Training")
    print("=" * 50)
    
    # Load model
    print(f"\nLoading model: {args.model}")
    model = YOLO(args.model)
    
    # Train
    print(f"\nStarting training for {args.epochs} epochs...")
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        patience=args.patience,
        save=True,
        save_period=10,
        plots=True,
        amp=False,
        workers=args.workers
    )
    
    print("\n" + "=" * 50)
    print("Training completed!")
    print("=" * 50)
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Train YOLOv8 for Steel Defect Detection')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                        choices=['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt'],
                        help='Model to use for training')
    parser.add_argument('--data', type=str, default='data/data.yaml',
                        help='Path to data.yaml file')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=150,
                        help='Number of training epochs')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Image size for training')
    parser.add_argument('--batch', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--device', type=str, default='0',
                        help='Device to use (0 for GPU, cpu for CPU)')
    parser.add_argument('--patience', type=int, default=50,
                        help='Early stopping patience')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of dataloader workers')
    
    # Output arguments
    parser.add_argument('--project', type=str, default='runs/train',
                        help='Project directory')
    parser.add_argument('--name', type=str, default='steel_defect',
                        help='Experiment name')
    
    args = parser.parse_args()
    
    # Run training
    train(args)


if __name__ == '__main__':
    main()
