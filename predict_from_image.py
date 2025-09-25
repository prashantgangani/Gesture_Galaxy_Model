
import sys
import os
import pickle
import cv2
import mediapipe as mp
import numpy as np
import traceback

def predict_from_image(image_path):
    try:
        # Load the model
        model_path = './model.p'
        if not os.path.exists(model_path):
            return "Model file not found"
        
        model_dict = pickle.load(open(model_path, 'rb'))
        model = model_dict['model']
        
        # Initialize MediaPipe hands with lower confidence for better detection
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.1, min_tracking_confidence=0.1)
        
        # Labels dictionary
        labels_dict = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 
                       10: 'a', 11: 'b', 12: 'c', 13: 'd', 14: 'e', 15: 'f', 16: 'g', 17: 'h', 18: 'i', 19: 'j', 
                       20: 'k', 21: 'l', 22: 'm', 23: 'n', 24: 'o', 25: 'p', 26: 'q', 27: 'r', 28: 's', 29: 't', 
                       30: 'u', 31: 'v', 32: 'w', 33: 'x', 34: 'y', 35: 'z'}
        
        # Read the image
        img = cv2.imread(image_path)
        if img is None:
            return "No image found"
        
        # Resize image to match training data
        img = cv2.resize(img, (640, 480))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        H, W, _ = img.shape
        
        # Process with MediaPipe
        results = hands.process(img_rgb)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                data_aux = []
                x_ = []
                y_ = []
                
                # Extract landmarks
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)
                
                # Normalize landmarks
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))
                
                # Pad data if necessary
                if len(data_aux) < 84:
                    data_aux = np.pad(data_aux, (0, 84 - len(data_aux)), 'constant')
                
                # Make prediction
                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = prediction[0]  # Use prediction directly since it's already the character
                
                # Calculate bounding box coordinates
                x1 = int(min(x_) * W) 
                y1 = int(min(y_) * H) 
                x2 = int(max(x_) * W) 
                y2 = int(max(y_) * H)
                
                # Draw hand landmarks (21 points)
                for i, (x, y) in enumerate(zip(x_, y_)):
                    cx, cy = int(x * W), int(y * H)
                    cv2.circle(img, (cx, cy), 3, (0, 255, 0), -1)  # Green dots for landmarks
                    cv2.circle(img, (cx, cy), 5, (0, 0, 0), 2)    # Black border around dots
                
                # Draw connections between landmarks
                connections = [
                    [0, 1], [1, 2], [2, 3], [3, 4],  # Thumb
                    [0, 5], [5, 6], [6, 7], [7, 8],  # Index finger
                    [0, 9], [9, 10], [10, 11], [11, 12],  # Middle finger
                    [0, 13], [13, 14], [14, 15], [15, 16],  # Ring finger
                    [0, 17], [17, 18], [18, 19], [19, 20],  # Pinky
                    [5, 9], [9, 13], [13, 17]  # Palm connections
                ]
                
                for connection in connections:
                    if connection[0] < len(x_) and connection[1] < len(x_):
                        pt1 = (int(x_[connection[0]] * W), int(y_[connection[0]] * H))
                        pt2 = (int(x_[connection[1]] * W), int(y_[connection[1]] * H))
                        cv2.line(img, pt1, pt2, (255, 0, 0), 2)  # Blue lines for connections
                
                # Draw bounding box and prediction on image
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 4)  # Red border
                cv2.putText(img, predicted_character, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3, cv2.LINE_AA)
                
                # Save the annotated image and convert to base64
                output_path = image_path.replace('.jpg', '_annotated.jpg')
                cv2.imwrite(output_path, img)
                
                # Convert annotated image to base64
                import base64
                with open(output_path, 'rb') as img_file:
                    img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
                
                return f"{predicted_character}|{img_base64}"
        
        return "No hand detected"
        
    except Exception as e:
        return f"Error: {str(e)}\nTraceback: {traceback.format_exc()}"

if __name__ == "__main__":
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        result = predict_from_image(image_path)
        print(result)
    else:
        print("No image path provided")
