from flask import Flask, request, jsonify
import base64
from io import BytesIO
from PIL import Image
import numpy as np

import pickle
import cv2
import mediapipe as mp

# Load model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Setup mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5',6: '6', 7: '7', 8: '8', 9:'9', 10: 'a', 11: 'b', 12: 'c', 13:'d',14: 'e', 15: 'f', 16: 'g', 17: 'h', 18: 'i', 19: 'j' , 20: 'k' , 21: 'l' ,22: 'm', 23: 'n', 24: 'o', 25: 'p', 26: 'q', 27: 'r',28: 's',29: 't',30: 'u',31: 'v', 32: 'w', 33: 'x',34: 'y',35: 'z',}

app = Flask(__name__)

# Dummy predict function (replace with your model logic)
def predict_from_image(image):
    # Convert PIL Image to OpenCV format
    import time
    image = image.convert('RGB')
    open_cv_image = np.array(image)
    # Convert RGB to BGR (OpenCV default)
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    # Resize to 640x480 for consistency
    open_cv_image = cv2.resize(open_cv_image, (640, 480))
    # Save the received image for debugging
    import os
    temp_dir = os.path.join(os.path.dirname(__file__), 'temp')
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    cv2.imwrite(os.path.join(temp_dir, f"received_{int(time.time()*1000)}.jpg"), open_cv_image)
    H, W, _ = open_cv_image.shape
    frame_rgb = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            x_ = []
            y_ = []
            data_aux = []
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))
            if len(data_aux) < 84:
                data_aux = np.pad(data_aux, (0, 84 - len(data_aux)), 'constant')
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = prediction[0]
            return {'prediction': str(predicted_character)}
    return {'prediction': None, 'error': 'No hand detected'}

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'error': 'No image provided'}), 400
    try:
        # Decode base64 image
        image_data = base64.b64decode(data['image'].split(',')[-1])
        image = Image.open(BytesIO(image_data))
        # Optionally preprocess image here
        result = predict_from_image(image)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
