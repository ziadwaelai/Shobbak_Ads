from flask import Flask, request, jsonify
import cv2
import numpy as np
import os

app = Flask(__name__)

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

@app.route('/detect_color', methods=['POST'])
def detect_color():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files['image']
    print(file.filename)
    image_path =file.filename
    file.save(image_path)

    image = cv2.imread(image_path)

    if is_black_image(image):
        return jsonify({"prediction": 0, "reason": "black image"})
    elif is_red_image(image):
        return jsonify({"prediction": 0, "reason": "red image"})
    else:
        return jsonify({"prediction": 1})
    # os.remove(image_path)
if __name__ == '__main__':
    app.run(debug=True,port=5000)

# curl -X POST -F "image=@D:/shobbak/AI/red.jpeg" http://127.0.0.1:5000/detect_color