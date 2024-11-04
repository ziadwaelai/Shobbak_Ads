from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
model = None

def load_model():
    global model
    model = tf.keras.models.load_model("watermark_detector_sequential_mobilenetv2.keras")
    print("Model loaded successfully")

def preprocess_image(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

def predict_watermark(img_path):
    img_array = preprocess_image(img_path)
    prediction = model.predict(img_array)
    return int(prediction[0][0] > 0.5)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No image selected for uploading"}), 400
    
    # Save the uploaded image
    file_path = secure_filename(file.filename)
    file.save(file_path)
    
    # Perform prediction
    try:
        result = predict_watermark(file_path)
        os.remove(file_path)  # Clean up the saved file
        return jsonify({"prediction": "Watermark" if result == 1 else "No Watermark"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    load_model()
    app.run(port=5000)


# curl -X POST -F "image=@D:/Users/ziad.abdlhamed/Downloads/webp (1).jpg" http://localhost:5000/predict
