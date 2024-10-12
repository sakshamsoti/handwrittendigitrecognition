from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import base64

# Load the trained model
model = tf.keras.models.load_model('digit_classifier_model.h5')

# Initialize the Flask app
app = Flask(__name__)

# Route for home page
@app.route('/')
def index():
    return render_template('index.html')

# Route for model prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get the image from the POST request
    data = request.get_json()
    image_data = data['image']
    image = Image.open(io.BytesIO(base64.b64decode(image_data))).convert('L')
    image = image.resize((28, 28))
    
    # Preprocess the image for the model
    image = np.array(image).reshape(1, 28, 28, 1).astype('float32') / 255.0
    
    # Make prediction
    prediction = model.predict(image)
    predicted_digit = np.argmax(prediction)
    
    return jsonify({'prediction': int(predicted_digit)})

if __name__ == '__main__':
    app.run(debug=True)