import cv2
import numpy as np
from ultralytics import YOLO
import os
from datetime import datetime

class CameraLicensePlateDetector:
    def __init__(self, model_path="yolo_training/vnlp_model7/weights/best.pt", confidence_threshold=0.5, crop_threshold=0.9):
        """
        Initialize the camera license plate detector
        
        Args:
            model_path: Path to the trained YOLO model
            confidence_threshold: Minimum confidence for showing detections
            crop_threshold: Minimum confidence for cropping detected plates
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.crop_threshold = crop_threshold
        self.crop_counter = 0
        
        # Create directory for saved crops
        self.crop_dir = "detected_plates"
        if not os.path.exists(self.crop_dir):
            os.makedirs(self.crop_dir)
        
        # Load the trained model
        print(f"Loading model from: {model_path}")
        self.model = YOLO(model_path)
        print("Model loaded successfully!")
        
        # Initialize camera
        self.cap = None
        
    def start_camera(self, camera_id=0):
        """Start the camera capture"""
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise Exception(f"Could not open camera {camera_id}")
        
        # Set camera properties for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
    def crop_plate(self, frame, box, confidence):
        """
        Crop the detected license plate from the frame
        
        Args:
            frame: Original frame
            box: Detection bounding box
            confidence: Detection confidence
        """
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        
        # Add some padding around the detected plate
        padding = 10
        h, w = frame.shape[:2]
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)
        
        # Crop the plate
        cropped_plate = frame[y1:y2, x1:x2]
        
        # Save the cropped plate
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"plate_{timestamp}_{self.crop_counter:04d}_conf{confidence:.3f}.jpg"
        filepath = os.path.join(self.crop_dir, filename)
        
        cv2.imwrite(filepath, cropped_plate)
        self.crop_counter += 1
        
        print(f"High confidence plate detected! Saved: {filename} (conf: {confidence:.3f})")
        return cropped_plate, filepath
    
    def draw_detections(self, frame, results):
        """
        Draw bounding boxes and labels on the frame
        
        Args:
            frame: Original frame
            results: YOLO detection results
            
        Returns:
            frame: Frame with drawn detections
        """
        cropped_plates = []
        
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    # Get box coordinates and confidence
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    confidence = box.conf[0].item()
                    
                    # Only show detections above threshold
                    if confidence >= self.confidence_threshold:
                        # Choose color based on confidence
                        if confidence >= self.crop_threshold:
                            color = (0, 255, 0)  # Green for high confidence
                            thickness = 3
                            
                            # Crop the plate if confidence is high enough
                            cropped_plate, filepath = self.crop_plate(frame, box, confidence)
                            cropped_plates.append((cropped_plate, filepath, confidence))
                            
                        else:
                            color = (0, 255, 255)  # Yellow for medium confidence
                            thickness = 2
                        
                        # Draw bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                        
                        # Draw confidence label
                        label = f"Plate: {confidence:.2f}"
                        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                                    (x1 + label_size[0], y1), color, -1)
                        cv2.putText(frame, label, (x1, y1 - 5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        return frame, cropped_plates
    
    def run_detection(self):
        """
        Main detection loop
        """
        if self.cap is None:
            self.start_camera()
        
        print("Starting license plate detection...")
        print(f"Detection threshold: {self.confidence_threshold}")
        print(f"Crop threshold: {self.crop_threshold}")
        print("Press 'q' to quit, 's' to save current frame, 'c' to clear crop counter")
        
        frame_count = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            frame_count += 1
            
            # Run detection every frame (you can skip frames for better performance)
            if frame_count % 1 == 0:  # Process every frame
                results = self.model(frame, conf=self.confidence_threshold, verbose=False)
                
                # Draw detections
                frame, cropped_plates = self.draw_detections(frame, results)
                
                # Show any newly cropped plates in separate windows
                for i, (cropped_plate, filepath, conf) in enumerate(cropped_plates):
                    if cropped_plate.shape[0] > 20 and cropped_plate.shape[1] > 20:  # Only show if big enough
                        # Resize cropped plate for better visibility
                        height, width = cropped_plate.shape[:2]
                        if width < 200:
                            scale = 200 / width
                            new_width = int(width * scale)
                            new_height = int(height * scale)
                            cropped_plate = cv2.resize(cropped_plate, (new_width, new_height))
                        
                        cv2.imshow(f"Detected Plate - {conf:.3f}", cropped_plate)
            
            # Add frame info
            info_text = f"Frame: {frame_count} | Crops saved: {self.crop_counter}"
            cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Add threshold info
            threshold_text = f"Detection: {self.confidence_threshold} | Crop: {self.crop_threshold}"
            cv2.putText(frame, threshold_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show the frame
            cv2.imshow('License Plate Detection', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current frame
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"frame_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Frame saved as: {filename}")
            elif key == ord('c'):
                # Clear crop counter
                self.crop_counter = 0
                print("Crop counter reset")
            elif key == ord('+') or key == ord('='):
                # Increase crop threshold
                self.crop_threshold = min(1.0, self.crop_threshold + 0.05)
                print(f"Crop threshold: {self.crop_threshold:.2f}")
            elif key == ord('-'):
                # Decrease crop threshold
                self.crop_threshold = max(0.1, self.crop_threshold - 0.05)
                print(f"Crop threshold: {self.crop_threshold:.2f}")
    
    def cleanup(self):
        """Clean up resources"""
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        print(f"Detection complete. {self.crop_counter} plates saved to '{self.crop_dir}' folder.")

def main():
    """Main function to run the camera detector"""
    try:
        # Create detector instance
        detector = CameraLicensePlateDetector(
            model_path="yolo_training/vnlp_model7/weights/best.pt",
            confidence_threshold=0.3,  # Show detections above 30%
            crop_threshold=0.88         # Only crop/save plates above 88%
        )
        
        # Start detection
        detector.run_detection()
        
    except KeyboardInterrupt:
        print("\nDetection interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        detector.cleanup()

if __name__ == "__main__":
    main()
