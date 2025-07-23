from ultralytics import YOLO
import cv2
import os

def predict_license_plate(model_path, image_path, confidence_threshold=0.5):
    """
    Predict license plate in an image using trained YOLO model
    
    Args:
        model_path: Path to the trained YOLO model (.pt file)
        image_path: Path to the input image
        confidence_threshold: Minimum confidence score for detections
    
    Returns:
        Detection results
    """
    # Load the trained model
    model = YOLO(model_path)
    
    # Perform inference
    results = model(image_path, conf=confidence_threshold)
    
    # Process results
    for r in results:
        # Get image dimensions
        img = cv2.imread(image_path)
        h, w = img.shape[:2]
        
        print(f"\nImage: {image_path}")
        print(f"Image size: {w}x{h}")
        print(f"Number of detections: {len(r.boxes) if r.boxes is not None else 0}")
        
        if r.boxes is not None:
            for i, box in enumerate(r.boxes):
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = box.conf[0].item()
                class_id = box.cls[0].item()
                
                print(f"  Detection {i+1}:")
                print(f"    Confidence: {confidence:.3f}")
                print(f"    Bounding box: ({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})")
                print(f"    Box size: {x2-x1:.1f} x {y2-y1:.1f}")
    
    return results

def batch_predict(model_path, image_folder, output_folder="predictions", confidence_threshold=0.5):
    """
    Predict license plates in multiple images
    
    Args:
        model_path: Path to the trained YOLO model
        image_folder: Folder containing images to process
        output_folder: Folder to save prediction results
        confidence_threshold: Minimum confidence score
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Load model
    model = YOLO(model_path)
    
    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    
    # Process all images in folder
    processed_count = 0
    for filename in os.listdir(image_folder):
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            image_path = os.path.join(image_folder, filename)
            
            # Perform prediction
            results = model(image_path, conf=confidence_threshold)
            
            # Save annotated image
            output_path = os.path.join(output_folder, f"pred_{filename}")
            results[0].save(output_path)
            
            processed_count += 1
            print(f"Processed: {filename} -> {output_path}")
    
    print(f"\nBatch prediction completed! Processed {processed_count} images.")
    print(f"Results saved in: {output_folder}")

if __name__ == "__main__":
    # Example usage
    
    # Path to your trained model (update this after training)
    MODEL_PATH = "yolo_training/vnlp_model7/weights/best.pt"
    
    # Test on single image
    print("=== Single Image Prediction ===")
    if os.path.exists(MODEL_PATH):
        test_image = "Resources/Photos/Validate/n0_multiple.png"
        if os.path.exists(test_image):
            results = predict_license_plate(MODEL_PATH, test_image, confidence_threshold=0.3)
            results[0].save("single_prediction.jpg")
            print("Prediction saved as 'single_prediction.jpg'")
        else:
            print(f"Test image not found: {test_image}")
    else:
        print(f"Model not found: {MODEL_PATH}")
        print("Please train the model first using plate_train.py")
    
    # Test on multiple images from test set
    print("\n=== Batch Prediction ===")
    if os.path.exists(MODEL_PATH):
        test_folder = "VNLP yolov8/test/images"
        if os.path.exists(test_folder):
            # Process first 5 images as example
            batch_predict(MODEL_PATH, test_folder, "batch_predictions", confidence_threshold=0.3)
        else:
            print(f"Test folder not found: {test_folder}")
    else:
        print("Model not found. Please train the model first.")
    
    # BONUS: Test on your own custom images
    print("\n=== Custom Image Testing ===")
    print("To test your own images:")
    print("1. Put your images in a folder (e.g., 'my_images/')")
    print("2. Run: batch_predict(MODEL_PATH, 'my_images/', 'my_results/')")
    print("3. Or test single image: predict_license_plate(MODEL_PATH, 'path/to/image.jpg')")
