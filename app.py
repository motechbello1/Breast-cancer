import tensorflow as tf
import numpy as np
from PIL import Image
import streamlit as st
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Load the pre-trained model
model = tf.keras.models.load_model('./model_eff.h5')

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

# Streamlit
st.title('Breast Cancer Detection Model')

upload = st.file_uploader('Upload an image to predict:', type=['jpg', 'png', 'jpeg'])

if upload:
    img = Image.open(upload)
    st.image(img, caption='Uploaded image', use_column_width=True)

    image = preprocess_image(upload)
    prediction = make_prediction(model, image)

    st.write('Prediction:', prediction.title())
    if prediction != "Unknown Image Detected":
        st.write('Please consult with your doctor for further examination.')