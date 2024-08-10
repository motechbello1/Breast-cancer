import os
from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model
model = tf.keras.models.load_model(r"C:\\Users\\player1\\Downloads\\Mr joel\\model_eff.h5")

def preprocess_image(image):
    image = Image.open(image)
    image = image.resize((224, 224))
    img_array = img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

def make_prediction(model, image):
    class_names = ['benign', 'malignant']
    predictions = model.predict(image)
    max_prob = np.max(predictions)

    if max_prob < 0.8:
        return "Unknown Image Detected"
    else:
        predicted_class = int(predictions > 0.5)
        return class_names[predicted_class]

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file part"
        
        file = request.files["file"]
        if file.filename == "":
            return "No selected file"
        
        if file:
            image = preprocess_image(file)
            prediction = make_prediction(model, image)
            return render_template("result.html", prediction=prediction.title())
    
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
