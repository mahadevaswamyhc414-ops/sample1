import os
import io
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, redirect, url_for
from PIL import Image

app = Flask(__name__)
model = None

# Class names for CIFAR-10
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

def load_trained_model():
    global model
    model_path = 'cifar10_model.keras'
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}...")
        try:
             model = tf.keras.models.load_model(model_path)
             print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            model = None
    else:
        print(f"Model file {model_path} not found. Please train the model first.")
        model = None

def prepare_image(image, target_size=(32, 32)):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = np.array(image)
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            try:
                # Read image
                img_bytes = file.read()
                image = Image.open(io.BytesIO(img_bytes))
                
                # Predict
                if model:
                    processed_image = prepare_image(image)
                    prediction = model.predict(processed_image)
                    class_idx = np.argmax(prediction, axis=1)[0]
                    class_name = CLASS_NAMES[class_idx]
                    confidence = float(prediction[0][class_idx])
                    
                    return render_template('result.html', 
                                           class_name=class_name, 
                                           confidence=f"{confidence:.2%}")
                else:
                     return "Model not loaded. Please train the model first."
            except Exception as e:
                return f"Error processing image: {e}"
    return render_template('index.html')

if __name__ == '__main__':
    load_trained_model()
    app.run(debug=True, host='0.0.0.0', port=5000)
