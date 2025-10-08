import cv2
import numpy as np
import pickle
import os
from datetime import datetime
import hashlib
from collections import Counter
import requests

class FoodHealthDetector:
    def __init__(self):
        self.data_file = 'food_health_data.pkl'
        self.training_data = []
        # ESP32 camera connection parameters
        self.esp32_camera_url = "http://192.168.1.104"  # Your ESP32 camera IP
        self.esp32_stream_url = "http://192.168.1.104:81/stream"  # Common ESP32 camera stream URL
        self.food_detected = False
        self.food_contours = []  # Changed to store multiple contours
        self.cap = None
        self.load_data()
        self.init_esp32_camera()
   
    def init_esp32_camera(self):
        """Initialize connection to ESP32 camera"""
        try:
            # Try to connect to the ESP32 camera
            print(f"Attempting to connect to ESP32 camera at {self.esp32_camera_url}")
            
            # For ESP32 cameras, we typically use OpenCV's VideoCapture
            # Try different common URLs for ESP32 camera streams
            possible_urls = [
                self.esp32_stream_url,
                "http://192.168.1.104:81/stream",  # Updated with your IP
                "http://192.168.4.1:81/stream",  # Common ESP32 AP mode IP
                "http://esp32cam.local:81/stream",  # mDNS address if supported
            ]
            
            for url in possible_urls:
                self.cap = cv2.VideoCapture(url)
                if self.cap.isOpened():
                    print(f"Connected to ESP32 camera at: {url}")
                    # Set some properties for better performance
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    self.cap.set(cv2.CAP_PROP_FPS, 10)
                    return
                
            # If no URL worked, try with index 0 (USB camera as fallback)
            print("Could not connect to ESP32 camera, trying default camera...")
            self.cap = cv2.VideoCapture(0)
            if self.cap.isOpened():
                print("Using default camera as fallback")
            else:
                print("Failed to initialize any camera")
                
        except Exception as e:
            print(f"Error initializing camera: {e}")
            # Fallback to default camera
            self.cap = cv2.VideoCapture(0)
   
    def load_data(self):
        """Load existing training data if available"""
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'rb') as f:
                    self.training_data = pickle.load(f)
                print(f"Loaded {len(self.training_data)} existing food samples")
            except:
                print("Could not load existing data, starting fresh")
                self.training_data = []
   
    def save_data(self):
        """Save training data to file"""
        with open(self.data_file, 'wb') as f:
            pickle.dump(self.training_data, f)
        print(f"Data saved! Total samples: {len(self.training_data)}")
   
    def extract_features(self, image):
        """Extract simple features from the image for classification"""
        # Resize for consistent feature extraction
        resized = cv2.resize(image, (100, 100))
       
        # Extract color histogram features
        hist_b = cv2.calcHist([resized], [0], None, [32], [0, 256])
        hist_g = cv2.calcHist([resized], [1], None, [32], [0, 256])
        hist_r = cv2.calcHist([resized], [2], None, [32], [0, 256])
       
        # Normalize histograms
        hist_b = hist_b.flatten() / hist_b.sum()
        hist_g = hist_g.flatten() / hist_g.sum()
        hist_r = hist_r.flatten() / hist_r.sum()
       
        # Combine features
        features = np.concatenate([hist_b, hist_g, hist_r])
       
        # Add average color features
        avg_color = resized.mean(axis=(0, 1)) / 255.0
        features = np.concatenate([features, avg_color])
       
        return features
   
    def predict_health(self, image):
        """Predict if food is healthy based on training data"""
        if len(self.training_data) < 2:
            return "Need more training data", 0.0
       
        # Extract features from current image
        current_features = self.extract_features(image)
       
        # Find k-nearest neighbors (simple KNN approach)
        k = min(5, len(self.training_data))
        distances = []
       
        for data in self.training_data:
            stored_features = data['features']
            label = data['label']
            # Calculate Euclidean distance
            dist = np.linalg.norm(current_features - stored_features)
            distances.append((dist, label))
       
        # Sort by distance and get k nearest
        distances.sort(key=lambda x: x[0])
        k_nearest = distances[:k]
       
        # Vote based on k nearest neighbors
        votes = [label for _, label in k_nearest]
        vote_counts = Counter(votes)
       
        # Get prediction and confidence
        prediction = vote_counts.most_common(1)[0][0]
        confidence = vote_counts[prediction] / k
       
        return prediction, confidence
    
    def detect_food(self, frame, min_area=1000):
        """Detect if there are multiple food items in the frame using color and contour analysis"""
        # Convert to HSV color space for better color segmentation
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define color ranges for common food colors (browns, yellows, greens, reds)
        lower_brown = np.array([10, 50, 20])
        upper_brown = np.array([20, 255, 200])
        
        lower_yellow = np.array([20, 50, 50])
        upper_yellow = np.array([35, 255, 255])
        
        lower_green = np.array([35, 50, 50])
        upper_green = np.array([85, 255, 255])
        
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        
        # Create masks for each color range
        mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        
        # Combine all masks
        combined_mask = cv2.bitwise_or(mask_brown, mask_yellow)
        combined_mask = cv2.bitwise_or(combined_mask, mask_green)
        combined_mask = cv2.bitwise_or(combined_mask, mask_red)
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours in the mask
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area to find potential food items
        food_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:  # Minimum area threshold
                food_contours.append(contour)
        
        # Update the state
        self.food_detected = len(food_contours) > 0
        self.food_contours = food_contours
        
        return self.food_detected, food_contours
    
    def get_esp32_camera_frame(self):
        """Get frame from ESP32 camera using OpenCV VideoCapture"""
        try:
            if self.cap is None or not self.cap.isOpened():
                self.init_esp32_camera()
                
            ret, frame = self.cap.read()
            if ret:
                return True, frame
            else:
                print("Failed to read frame from camera")
                # Try to reinitialize the camera
                self.init_esp32_camera()
                return False, None
                
        except Exception as e:
            print(f"Error accessing camera: {e}")
            return False, None

    def training_mode(self):
        """Mode for labeling food as healthy or unhealthy - supports multiple items"""
        print("\n=== TRAINING MODE ===")
        print("Press 'h' to mark selected item as HEALTHY")
        print("Press 'u' to mark selected item as UNHEALTHY")
        print("Press 'a' to mark ALL detected items with the same label")
        print("Press 's' to skip current frame")
        print("Press 'q' to exit training mode")
        print("Press SPACE to capture frame for labeling")
        print("Use LEFT/RIGHT arrows to cycle through detected items")
       
        print("Using ESP32 Camera")
        
        selected_item_idx = 0  # Track which item is currently selected
       
        while True:
            # Get frame from ESP32 camera
            success, frame = self.get_esp32_camera_frame()
            
            if not success:
                print("Failed to get frame from camera. Check connection.")
                cv2.waitKey(1000)
                continue
            
            # Detect food in the frame
            food_found, food_contours = self.detect_food(frame)
           
            # Display frame
            display_frame = frame.copy()
            
            # Draw bounding boxes if food is detected
            if food_found:
                for i, contour in enumerate(food_contours):
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Highlight the selected item
                    if i == selected_item_idx:
                        color = (0, 255, 255)  # Yellow for selected
                        thickness = 3
                        cv2.putText(display_frame, f"Selected #{i+1}", (x, y-15), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    else:
                        color = (0, 255, 0)  # Green for others
                        thickness = 2
                    
                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, thickness)
                    cv2.putText(display_frame, f"Item {i+1}", (x, y-5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            cv2.putText(display_frame, "TRAINING MODE - Press SPACE to capture",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Samples collected: {len(self.training_data)}",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(display_frame, f"Detected items: {len(food_contours)}",
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(display_frame, "Camera: ESP32",
                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
           
            cv2.imshow('Food Health Detector - Training', display_frame)
           
            key = cv2.waitKey(1) & 0xFF
           
            if key == ord(' '):  # Space to capture
                # Only capture if food is detected
                if not food_found:
                    print("No food detected in the frame. Please try again.")
                    continue
                    
                # Freeze frame for labeling
                frozen_frame = frame.copy()
                
                # Draw all contours
                for i, contour in enumerate(food_contours):
                    x, y, w, h = cv2.boundingRect(contour)
                    if i == selected_item_idx:
                        color = (0, 255, 255)  # Yellow for selected
                        thickness = 3
                    else:
                        color = (0, 255, 0)  # Green for others
                        thickness = 2
                    cv2.rectangle(frozen_frame, (x, y), (x+w, y+h), color, thickness)
                    cv2.putText(frozen_frame, f"Item {i+1}", (x, y-5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                cv2.putText(frozen_frame, "CAPTURED! Label options:",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(frozen_frame, "'h'=Healthy (selected), 'u'=Unhealthy (selected)",
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                cv2.putText(frozen_frame, "'a'=All items same label, 's'=Skip",
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                cv2.imshow('Food Health Detector - Training', frozen_frame)
               
                # Wait for label
                while True:
                    label_key = cv2.waitKey(0) & 0xFF
                   
                    if label_key == ord('h'):  # Label selected as healthy
                        self._label_food_item(frame, food_contours, selected_item_idx, "HEALTHY")
                        break
                   
                    elif label_key == ord('u'):  # Label selected as unhealthy
                        self._label_food_item(frame, food_contours, selected_item_idx, "UNHEALTHY")
                        break
                    
                    elif label_key == ord('a'):  # Label all with the same label
                        # Ask for the label to apply to all
                        cv2.putText(frozen_frame, "Apply to ALL: 'h'=Healthy, 'u'=Unhealthy",
                                   (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                        cv2.imshow('Food Health Detector - Training', frozen_frame)
                        
                        all_label_key = cv2.waitKey(0) & 0xFF
                        if all_label_key == ord('h'):
                            for i in range(len(food_contours)):
                                self._label_food_item(frame, food_contours, i, "HEALTHY")
                            break
                        elif all_label_key == ord('u'):
                            for i in range(len(food_contours)):
                                self._label_food_item(frame, food_contours, i, "UNHEALTHY")
                            break
                   
                    elif label_key == ord('s'):
                        print("↳ Skipped frame")
                        break
            
            # Navigation between items
            elif key == 81:  # Left arrow
                if food_found and len(food_contours) > 0:
                    selected_item_idx = (selected_item_idx - 1) % len(food_contours)
            
            elif key == 83:  # Right arrow
                if food_found and len(food_contours) > 0:
                    selected_item_idx = (selected_item_idx + 1) % len(food_contours)
           
            elif key == ord('q'):
                break
       
        cv2.destroyAllWindows()
    
    def _label_food_item(self, frame, contours, index, label):
        """Helper method to label a specific food item"""
        x, y, w, h = cv2.boundingRect(contours[index])
        food_region = frame[y:y+h, x:x+w]
        features = self.extract_features(food_region)
        self.training_data.append({
            'features': features,
            'label': label,
            'timestamp': datetime.now().isoformat()
        })
        print(f"✓ Labeled item {index+1} as {label}. Total samples: {len(self.training_data)}")
        self.save_data()
    
    def detection_mode(self):
        """Mode for detecting if food is healthy or unhealthy - supports multiple items"""
        print("\n=== DETECTION MODE ===")
        print("Press 'q' to exit detection mode")
        print("Press 't' to switch to training mode")
        print("Using ESP32 Camera")
       
        if len(self.training_data) < 2:
            print("⚠️ WARNING: Need at least 2 training samples for detection to work!")
       
        while True:
            # Get frame from ESP32 camera
            success, frame = self.get_esp32_camera_frame()
            
            if not success:
                print("Failed to get frame from camera. Check connection.")
                cv2.waitKey(1000)
                continue
           
            display_frame = frame.copy()
            
            # Detect food in the frame
            food_found, food_contours = self.detect_food(frame)
            
            # Show predictions for all detected food items
            if food_found:
                for i, contour in enumerate(food_contours):
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Crop the image to the food region
                    food_region = frame[y:y+h, x:x+w]
                    
                    # Predict health status
                    if len(self.training_data) >= 2:
                        prediction, confidence = self.predict_health(food_region)
                       
                        # Choose color based on prediction
                        if prediction == "HEALTHY":
                            color = (0, 255, 0)  # Green
                            status = "✓ HEALTHY"
                        else:
                            color = (0, 0, 255)  # Red
                            status = "✗ UNHEALTHY"
                       
                        # Draw bounding box and prediction
                        cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 3)
                        cv2.putText(display_frame, f"Item {i+1}: {status}",
                                   (x, y-40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                        cv2.putText(display_frame, f"Conf: {confidence:.1%}",
                                   (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    else:
                        cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
                        cv2.putText(display_frame, f"Item {i+1}: Need more training!",
                                   (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            else:
                # No food detected, just show the camera feed
                cv2.putText(display_frame, "No food detected", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
           
            cv2.putText(display_frame, f"Training samples: {len(self.training_data)}",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(display_frame, f"Detected items: {len(food_contours)}",
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(display_frame, "Camera: ESP32",
                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
           
            cv2.imshow('Food Health Detector - Detection', display_frame)
           
            key = cv2.waitKey(1) & 0xFF
           
            if key == ord('q'):
                break
            elif key == ord('t'):
                cv2.destroyAllWindows()
                self.training_mode()
                break
       
        cv2.destroyAllWindows()
   
    def show_statistics(self):
        """Show statistics about the training data"""
        if not self.training_data:
            print("No training data available")
            return
       
        healthy_count = sum(1 for d in self.training_data if d['label'] == 'HEALTHY')
        unhealthy_count = len(self.training_data) - healthy_count
       
        print("\n=== TRAINING DATA STATISTICS ===")
        print(f"📊 Total samples: {len(self.training_data)}")
        print(f"✅ Healthy samples: {healthy_count}")
        print(f"❌ Unhealthy samples: {unhealthy_count}")
        print(f"📅 Last sample: {self.training_data[-1]['timestamp']}")
   
    def clear_data(self):
        """Clear all training data"""
        confirm = input("Are you sure you want to clear all training data? (yes/no): ")
        if confirm.lower() == 'yes':
            self.training_data = []
            self.save_data()
            print("🗑️ Training data cleared!")
        else:
            print("Clear operation cancelled")
   
    def run(self):
        """Main program loop"""
        print("\n=== FOOD HEALTH DETECTOR ===")
        print("📷 Using ESP32 Camera: http://192.168.1.104")
        print("This program uses your ESP32 camera to detect and classify food as healthy or unhealthy")
       
        while True:
            print("\n--- MAIN MENU ---")
            print("1. 🎓 Training Mode (Label food as healthy/unhealthy)")
            print("2. 🔍 Detection Mode (Detect if food is healthy/unhealthy)")
            print("3. 📈 Show Statistics")
            print("4. 🗑️ Clear All Data")
            print("5. 🚪 Exit")
           
            choice = input("\nEnter your choice (1-5): ")
           
            if choice == '1':
                self.training_mode()
            elif choice == '2':
                self.detection_mode()
            elif choice == '3':
                self.show_statistics()
            elif choice == '4':
                self.clear_data()
            elif choice == '5':
                print("👋 Goodbye!")
                # Release the camera
                if self.cap is not None:
                    self.cap.release()
                break
            else:
                print("❌ Invalid choice, please try again")

# Installation requirements check
def check_requirements():
    """Check if required packages are installed"""
    try:
        import cv2
        import requests
        print("✓ OpenCV is installed")
        print("✓ Requests library is installed")
        return True
    except ImportError as e:
        print(f"✗ Missing package: {e}")
        print("\nTo install required packages, run:")
        print("pip install opencv-python numpy requests")
        return False

if __name__ == "__main__":
    if check_requirements():
        detector = FoodHealthDetector()
        detector.run()
    else:
        print("\nPlease install the required packages first!")
