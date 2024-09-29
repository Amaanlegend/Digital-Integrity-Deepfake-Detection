import cv2
import numpy as np
from flask import Flask, render_template, request
import tensorflow as tf

app = Flask(__name__)

# Load pre-trained deepfake detection model
model = tf.keras.models.load_model('models/deepfake_detector.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    file = request.files['media']
    if file:
        # Read file as image/video
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        
        # Preprocess the image
        img = cv2.resize(img, (224, 224))
        img = np.expand_dims(img, axis=0)
        
        # Predict using model
        prediction = model.predict(img)
        
        # Generate a confidence score
        confidence = prediction[0][0]
        
        # Return results
        return f"Deepfake Confidence: {confidence:.2f}"
    
if __name__ == "__main__":
    app.run(debug=True)
