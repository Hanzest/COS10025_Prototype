import os
import pandas as pd
from pathlib import Path

def find_best_model():
    """
    Compare all YOLO training runs and find the best performing model
    """
    print("=== YOLO Training Results Comparison ===\n")
    
    training_dir = Path("yolo_training")
    if not training_dir.exists():
        print("❌ No training directory found!")
        return None
    
    best_model = None
    best_map50 = 0
    best_map50_95 = 0
    
    # Find all vnlp_model directories
    model_dirs = sorted([d for d in training_dir.iterdir() if d.is_dir() and d.name.startswith('vnlp_model')])
    
    if not model_dirs:
        print("❌ No training results found!")
        return None
    
    print(f"Found {len(model_dirs)} training runs:\n")
    
    for i, model_dir in enumerate(model_dirs, 1):
        print(f"📁 {model_dir.name}:")
        
        # Check if results.csv exists
        results_file = model_dir / "results.csv"
        weights_dir = model_dir / "weights"
        best_pt = weights_dir / "best.pt"
        
        if results_file.exists():
            try:
                # Read training results
                df = pd.read_csv(results_file)
                if len(df) > 0:
                    # Get best metrics from the training
                    final_row = df.iloc[-1]  # Last epoch
                    
                    epochs = len(df)
                    map50 = final_row.get('metrics/mAP50(B)', 0)
                    map50_95 = final_row.get('metrics/mAP50-95(B)', 0)
                    box_loss = final_row.get('val/box_loss', 0)
                    cls_loss = final_row.get('val/cls_loss', 0)
                    
                    print(f"   ✅ Completed: {epochs} epochs")
                    print(f"   📊 mAP50: {map50:.4f}")
                    print(f"   📊 mAP50-95: {map50_95:.4f}")
                    print(f"   📉 Box Loss: {box_loss:.4f}")
                    print(f"   📉 Cls Loss: {cls_loss:.4f}")
                    
                    # Check if this is the best model
                    if map50 > best_map50:
                        best_map50 = map50
                        best_map50_95 = map50_95
                        best_model = model_dir
                        
                else:
                    print("   ⚠️  Empty results fil/e")
            except Exception as e:
                print(f"   ❌ Error reading results: {e}")
        else:
            print("   ⚠️  No results.csv found")
        
        # Check model files
        if best_pt.exists():
            file_size = best_pt.stat().st_size / (1024*1024)  # MB
            print(f"   💾 Model file: best.pt ({file_size:.1f}MB)")
        else:
            print("   ❌ No best.pt found")
        
        print()
    
    # Summary
    if best_model:
        print("🏆 BEST MODEL FOUND:")
        print(f"   📁 Directory: {best_model.name}")
        print(f"   📊 mAP50: {best_map50:.4f}")
        print(f"   📊 mAP50-95: {best_map50_95:.4f}")
        print(f"   📂 Path: {best_model / 'weights' / 'best.pt'}")
        
        return str(best_model / 'weights' / 'best.pt')
    else:
        print("❌ No valid models found!")
        return None

def update_predict_script(best_model_path):
    """
    Update the predict_plates.py script with the best model path
    """
    if best_model_path:
        print(f"\n🔧 Updating predict_plates.py with best model:")
        print(f"   New MODEL_PATH: {best_model_path}")
        
        # Read current file
        with open('predict_plates.py', 'r') as f:
            content = f.read()
        
        # Replace MODEL_PATH
        old_line = 'MODEL_PATH = "yolo_training/vnlp_model7/weights/best.pt"'
        new_line = f'MODEL_PATH = "{best_model_path}"'
        
        if old_line in content:
            content = content.replace(old_line, new_line)
            
            # Write updated file
            with open('predict_plates.py', 'w') as f:
                f.write(content)
            
            print("   ✅ predict_plates.py updated!")
        else:
            print(f"   ⚠️  Could not find MODEL_PATH line to update")
            print(f"   📝 Please manually update MODEL_PATH to: {best_model_path}")

if __name__ == "__main__":
    best_model_path = find_best_model()
    
    if best_model_path:
        update_predict_script(best_model_path)
        
        print(f"\n🚀 Ready to predict! Run:")
        print(f"   python predict_plates.py")
    else:
        print("\n❌ No usable models found. Please check your training results.")
