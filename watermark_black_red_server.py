from flask import Flask, request, jsonify
import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image # type: ignore

app = Flask(__name__)

# Load the watermark detection model at startup
def load_model():
    model = tf.keras.models.load_model("watermark_detector_sequential_mobilenetv2_last_L2_datesetNewV2.keras")
    return model

watermark_model = load_model()

def preprocess_image(img_path, target_size=(225, 225)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

def predict_watermark(img_path):
    img_array = preprocess_image(img_path)
    prediction = watermark_model.predict(img_array)
    return np.argmax(prediction)

def is_black_image(image, black_threshold=30):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    avg_intensity = np.mean(gray)
    return avg_intensity < black_threshold

def is_red_image(image, red_threshold=0.3):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = mask1 + mask2

    red_ratio = np.sum(red_mask > 0) / (image.shape[0] * image.shape[1])
    return red_ratio > red_threshold

@app.route('/check_image', methods=['POST'])
def check_image():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400

        file = request.files['image']
        image_path = file.filename
        file.save(image_path)

        # Load the image with OpenCV
        image = cv2.imread(image_path)

        # Check if the image is black, red, or watermarked
        if is_black_image(image):
            result = {"prediction": 0, "reason": "black image"}
        elif is_red_image(image):
            result = {"prediction": 0, "reason": "red image"}
        elif predict_watermark(image_path) == 1:
            result = {"prediction": 0, "reason": "watermarked image"}
        else:
            result = {"prediction": 1}

        # Optionally delete the saved image after processing
        # os.remove(image_path)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)

# curl -X POST -F "image=@D:/shobbak/AI/red.jpeg" http://127.0.0.1:5000/check_image