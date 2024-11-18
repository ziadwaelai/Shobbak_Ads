from flask import Flask, request, jsonify
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing import image  # type: ignore
from werkzeug.utils import secure_filename
import tempfile
from PIL import Image

app = Flask(__name__)

# Load the watermark detection model at startup
watermark_model = load_model("watermark_detector_sequential_mobilenetv2_last_L2_datesetNewV2.keras")
from sentence_transformers import SentenceTransformer, util
clip_model = SentenceTransformer('sentence-transformers/clip-ViT-B-32')



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


def is_black_image(image, threshold=30):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.mean(gray) < threshold


def is_red_image(image, threshold=0.3):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    red_mask = (
        cv2.inRange(hsv, np.array([0, 50, 50]), np.array([10, 255, 255]))
        + cv2.inRange(hsv, np.array([170, 50, 50]), np.array([180, 255, 255]))
    )
    return np.sum(red_mask > 0) / (image.shape[0] * image.shape[1]) > threshold

def check_title(title):
    # Check if the title contains any inappropriate words
    inappropriate_words = ["spam", "scam", "fraud", "fake"]
    return any(word in title.lower() for word in inappropriate_words)

def check_description(description):
    # Check if the description contains any inappropriate words
    inappropriate_words = ["spam", "scam", "fraud", "fake"]
    return any(word in description.lower() for word in inappropriate_words)

def check_consistency_text_image(images, title, threshold=0.2):
    """
    Checks if all images have a cosine similarity score above the threshold with the title.
    
    Args:
        images (list): List of image file paths or objects.
        title (str): The text to compare against the images.
        threshold (float): The similarity threshold.

    Returns:
        int: 1 if all images meet or exceed the threshold, otherwise 0.
    """
    if not images:
        raise ValueError("The list of images is empty.")

    # Encode the title
    title_emb = clip_model.encode(title)

    for image_file in images:
        try:
            # Open and encode the image directly from FileStorage
            image = Image.open(image_file)
            img_emb = clip_model.encode(image)

            # Compute cosine similarity
            cos_score = util.cos_sim(img_emb, title_emb)[0][0].item()
            print(f"Similarity between title and image: {cos_score}")

            # Return 0 if the similarity is below the threshold
            if cos_score < threshold:
                print(f"Image failed the similarity check: {cos_score} < {threshold}")
                return 0
        except Exception as e:
            print(f"Error processing image: {e}")

    # If all images meet or exceed the threshold
    print("All images passed the similarity check.")
    return 1


@app.route('/submit_ad', methods=['POST'])
def submit_ad():
    try:
        # Extract metadata
        title = request.form.get('title')
        description = request.form.get('description')
        category = request.form.get('category')

        if not title or not description or not category:
            return jsonify({"prediction": 0, "reason": "Missing title, description, or category"}), 400

        # Extract images
        files = request.files.getlist('images')

        # Validate title
        if check_title(title):
            return jsonify({"prediction": 0, "reason": "Inappropriate title"}), 400
        
        # Validate description
        if check_description(description):
            return jsonify({"prediction": 0, "reason": "Inappropriate description"}), 400

        # Validate uploaded images
        if (not files or files[0].filename == "") and category != "job":
            return jsonify({"prediction": 0, "reason": "No images provided"}), 400
        
        # Check consistency between text and images
        if not check_consistency_text_image([file.filename for file in files], title):
            return jsonify({"prediction": 0, "reason": "Inconsistent title and images"}), 400
        
        if files[0].filename != "":
            for file in files:
                # Save the uploaded image to a temporary location
                temp_path = os.path.join(tempfile.gettempdir(), secure_filename(file.filename))
                file.save(temp_path)
                try:
                    # Validate image
                    image = cv2.imread(temp_path)
                    if image is None:
                        return jsonify({"prediction": 0, "reason": f"{file.filename} is invalid"}), 400

                    if is_black_image(image):
                        return jsonify({"prediction": 0, "reason": f"{file.filename} is a black image"}), 400

                    if is_red_image(image):
                        return jsonify({"prediction": 0, "reason": f"{file.filename} is a red image"}), 400

                    if predict_watermark(temp_path) == 1:
                        return jsonify({"prediction": 0, "reason": f"{file.filename} is watermarked"}), 400
                finally:
                    # Clean up temp file
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

        # All images are valid
        return jsonify({"prediction": 1}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)
