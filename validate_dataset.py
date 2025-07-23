import os
import yaml
from pathlib import Path

def validate_dataset(data_yaml_path):
    """
    Validate YOLO dataset structure and configuration
    
    Args:
        data_yaml_path: Path to the data.yaml file
    """
    print("=== Dataset Validation ===")
    
    # Read data.yaml
    with open(data_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    print(f"Data configuration:")
    print(f"  Classes: {data_config['nc']}")
    print(f"  Class names: {data_config['names']}")
    
    # Get dataset root directory
    yaml_dir = Path(data_yaml_path).parent
    
    # Check each split
    splits = ['train', 'val', 'test']
    total_stats = {}
    
    for split in splits:
        if split in data_config:
            # Resolve path relative to yaml file
            images_path = yaml_dir / data_config[split]
            labels_path = images_path.parent / 'labels'
            
            print(f"\n--- {split.upper()} SET ---")
            print(f"Images path: {images_path}")
            print(f"Labels path: {labels_path}")
            
            # Count files
            if images_path.exists():
                image_files = list(images_path.glob('*'))
                image_count = len([f for f in image_files if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
                print(f"Images found: {image_count}")
            else:
                image_count = 0
                print(f"⚠️  Images directory not found!")
            
            if labels_path.exists():
                label_files = list(labels_path.glob('*.txt'))
                label_count = len(label_files)
                print(f"Labels found: {label_count}")
                
                # Check label format
                if label_files:
                    sample_label = label_files[0]
                    try:
                        with open(sample_label, 'r') as f:
                            line = f.readline().strip()
                            if line:
                                parts = line.split()
                                print(f"Sample label format: class={parts[0]}, bbox=[{', '.join(parts[1:])}]")
                    except Exception as e:
                        print(f"⚠️  Error reading sample label: {e}")
            else:
                label_count = 0
                print(f"⚠️  Labels directory not found!")
            
            # Check if image and label counts match
            if image_count == label_count:
                print(f"✅ Image-label pairs: {image_count}")
            else:
                print(f"⚠️  Mismatch! Images: {image_count}, Labels: {label_count}")
            
            total_stats[split] = {'images': image_count, 'labels': label_count}
    
    # Summary
    print("\n=== SUMMARY ===")
    total_images = sum(stats['images'] for stats in total_stats.values())
    total_labels = sum(stats['labels'] for stats in total_stats.values())
    print(f"Total images: {total_images}")
    print(f"Total labels: {total_labels}")
    
    if total_images > 0 and total_labels > 0:
        print("✅ Dataset appears to be properly structured!")
        return True
    else:
        print("❌ Dataset has issues that need to be fixed!")
        return False

def check_sample_annotations(data_yaml_path, num_samples=5):
    """
    Display sample annotations to verify data quality
    """
    print("\n=== SAMPLE ANNOTATIONS ===")
    
    with open(data_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    yaml_dir = Path(data_yaml_path).parent
    train_labels_path = yaml_dir / 'train' / 'labels'
    
    if train_labels_path.exists():
        label_files = list(train_labels_path.glob('*.txt'))[:num_samples]
        
        for i, label_file in enumerate(label_files, 1):
            print(f"\nSample {i}: {label_file.name}")
            try:
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                    for j, line in enumerate(lines):
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id, x_center, y_center, width, height = parts[:5]
                            print(f"  Object {j+1}: class={class_id}, center=({x_center}, {y_center}), size=({width}, {height})")
                        else:
                            print(f"  ⚠️  Invalid annotation: {line.strip()}")
            except Exception as e:
                print(f"  ⚠️  Error reading file: {e}")

if __name__ == "__main__":
    # Validate the Vietnamese license plate dataset
    data_yaml_path = "VNLP yolov8/data.yaml"
    
    if os.path.exists(data_yaml_path):
        is_valid = validate_dataset(data_yaml_path)
        
        if is_valid:
            check_sample_annotations(data_yaml_path)
            print("\n✅ Dataset is ready for training!")
            print("Run 'python plate_train.py' to start training.")
        else:
            print("\n❌ Please fix dataset issues before training.")
    else:
        print(f"❌ Data configuration file not found: {data_yaml_path}")
        print("Please check your dataset structure.")
