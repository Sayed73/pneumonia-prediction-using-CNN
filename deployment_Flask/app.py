from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os

app = Flask(__name__)
model = load_model("pneumonia_prediction_model.h5")

# Define image preprocessing function
def preprocess_image(image, img_size):
    try:
        # Read image in grayscale and resize it
        img_arr = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
        resized_arr = cv2.resize(img_arr, (img_size, img_size))
        
        # Normalize the image
        normalized_img = resized_arr / 255.0

        # Reshape the image to match the input shape of the model
        processed_img = normalized_img.reshape(-1, img_size, img_size, 1)

        return processed_img
    except Exception as e:
        print(e)
        return None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            # Preprocess the uploaded image
            processed_image = preprocess_image(file, img_size=128)

            if processed_image is not None:
                # Make prediction using the loaded model
                prediction = model.predict(processed_image)[0]
                result = "Pneumonia" if prediction <= 0.5 else "Normal Lung"
                return render_template('index.html', prediction=result)
            else:
                return "Failed to process image"
        else:
            msg = "No file uploaded"
            return render_template('index.html', prediction=msg)

if __name__ == '__main__':
    app.run(debug=True)
