import os
import time
from pathlib import Path

def monitor_training(training_dir="yolo_training/vnlp_model"):
    """
    Monitor YOLO training progress by checking logs and results
    """
    print("=== YOLO Training Monitor ===")
    
    training_path = Path(training_dir)
    results_file = training_path / "results.csv"
    weights_dir = training_path / "weights"
    
    print(f"Monitoring training at: {training_path}")
    print("Press Ctrl+C to stop monitoring")
    print()
    
    last_epoch = -1
    
    try:
        while True:
            if results_file.exists():
                # Read the results CSV file
                with open(results_file, 'r') as f:
                    lines = f.readlines()
                
                if len(lines) > 1:  # Header + data
                    # Get the latest epoch data
                    latest_line = lines[-1].strip()
                    if latest_line:
                        data = latest_line.split(',')
                        epoch = int(float(data[0].strip()))
                        
                        if epoch > last_epoch:
                            last_epoch = epoch
                            
                            # Extract key metrics
                            try:
                                train_box_loss = float(data[1].strip())
                                train_cls_loss = float(data[2].strip())
                                train_dfl_loss = float(data[3].strip())
                                val_box_loss = float(data[5].strip()) if len(data) > 5 else 0
                                val_cls_loss = float(data[6].strip()) if len(data) > 6 else 0
                                val_dfl_loss = float(data[7].strip()) if len(data) > 7 else 0
                                map50 = float(data[8].strip()) if len(data) > 8 else 0
                                map50_95 = float(data[9].strip()) if len(data) > 9 else 0
                                
                                print(f"Epoch {epoch:2d}/50:")
                                print(f"  Train - Box: {train_box_loss:.4f}, Cls: {train_cls_loss:.4f}, DFL: {train_dfl_loss:.4f}")
                                print(f"  Val   - Box: {val_box_loss:.4f}, Cls: {val_cls_loss:.4f}, DFL: {val_dfl_loss:.4f}")
                                print(f"  mAP50: {map50:.4f}, mAP50-95: {map50_95:.4f}")
                                print()
                                
                            except (ValueError, IndexError):
                                print(f"Epoch {epoch}: Data parsing error")
            
            # Check for saved weights
            if weights_dir.exists():
                best_pt = weights_dir / "best.pt"
                last_pt = weights_dir / "last.pt"
                
                if best_pt.exists() and last_pt.exists():
                    best_size = best_pt.stat().st_size / (1024*1024)  # MB
                    last_size = last_pt.stat().st_size / (1024*1024)  # MB
                    print(f"Saved models: best.pt ({best_size:.1f}MB), last.pt ({last_size:.1f}MB)")
            
            time.sleep(10)  # Check every 10 seconds
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")
    except Exception as e:
        print(f"Error: {e}")

def check_training_status():
    """
    Quick check of training status
    """
    training_path = Path("yolo_training/vnlp_model")
    
    if not training_path.exists():
        print("❌ Training not started yet")
        return
    
    results_file = training_path / "results.csv"
    weights_dir = training_path / "weights"
    
    print("=== Training Status ===")
    
    if results_file.exists():
        with open(results_file, 'r') as f:
            lines = f.readlines()
        
        if len(lines) > 1:
            latest_line = lines[-1].strip()
            if latest_line:
                data = latest_line.split(',')
                epoch = int(float(data[0].strip()))
                print(f"✅ Training in progress: Epoch {epoch}/50")
                
                if len(data) > 8:
                    map50 = float(data[8].strip())
                    map50_95 = float(data[9].strip())
                    print(f"   Latest mAP50: {map50:.4f}")
                    print(f"   Latest mAP50-95: {map50_95:.4f}")
            else:
                print("🟡 Training started but no data yet")
        else:
            print("🟡 Training started but no data yet")
    else:
        print("🟡 Training started but results file not created yet")
    
    if weights_dir.exists():
        best_pt = weights_dir / "best.pt"
        if best_pt.exists():
            print(f"✅ Best model saved: {best_pt}")
        else:
            print("🟡 Best model not saved yet")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "monitor":
        monitor_training()
    else:
        check_training_status()
