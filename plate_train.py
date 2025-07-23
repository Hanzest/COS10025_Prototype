from ultralytics import YOLO
import os
import torch
import time

def main():
    print(f"Python version: {torch.__version__ if hasattr(torch, '__version__') else 'Unknown'}")
    print(f"PyTorch version: {torch.__version__}")

    # Load a pretrained YOLO model (recommended for training)
    model = YOLO("yolov8n.pt")

    # Check CUDA availability and set device
    if torch.cuda.is_available():
        device = 0  # Use GPU
        print(f"CUDA available! Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
        batch_size = 32
    else:
        device = 'cpu'  # Use CPU
        print("CUDA not available. Using CPU for training.")
        print("Note: Training will be slower on CPU. Consider installing CUDA-enabled PyTorch.")
        batch_size = 8  # Smaller batch for CPU

    # Train the model with the Vietnamese license plate dataset
    print("Starting training...")
    start_time = time.time()

    results = model.train(
        data="VNLP yolov8/data.yaml",  # Fixed path to data.yaml
        epochs=20 if device == 'cpu' else 50,  # Fewer epochs for CPU
        imgsz=640,  # Image size
        batch=batch_size,   # Adjust batch size based on device
        patience=10,  # Early stopping patience
        save=True,   # Save model checkpoints
        project="yolo_training",  # Project name
        name="vnlp_model",  # Run name
        device=device,  # Use detected device
        workers=0  # Fix multiprocessing issue on Windows
    )

    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time/60:.1f} minutes")

    # Save the best trained model
    best_model_path = "yolo_training/vnlp_model/weights/best.pt"
    print(f"Best model saved at: {best_model_path}")

    # Load the best trained model for inference
    trained_model = YOLO(best_model_path)

    # Test the trained model on validation image
    print("Testing trained model...")
    test_results = trained_model("Resources/Photos/Validate/n0_1111_real.png")

    # Save results
    test_results[0].save("prediction_result.jpg")
    print("Prediction saved as 'prediction_result.jpg'")

    # Print model performance metrics
    print("\nTraining completed!")
    print(f"Model metrics:")
    print(f"- mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
    print(f"- mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")

    # Test on a few more images from the test set
    print("\nTesting on additional images...")
    test_images = [
        "VNLP yolov8/test/images/CarLongPlate113_jpg.rf.b7aeea8b142c7063298bd4e2dde7cc18.jpg",
        "VNLP yolov8/test/images/CarLongPlate115_jpg.rf.209cd6d2686832a3d731907562a9607f.jpg",
        "VNLP yolov8/test/images/CarLongPlate130_jpg.rf.e58b8c9b2795a35aa2b1d79a31d307c5.jpg"
    ]

    for img_path in test_images:
        if os.path.exists(img_path):
            result = trained_model(img_path)
            output_name = f"prediction_{os.path.basename(img_path)}"
            result[0].save(output_name)
            print(f"Prediction saved as '{output_name}'")

if __name__ == '__main__':
    main()

