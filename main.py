import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_model.h5")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(64, 64))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # Return index of max element

# Custom CSS
st.markdown(
    """
    <style>
    .main-header {
        font-size: 48px;
        color: #2E86C1;
        text-align: center;
        font-weight: bold;
        margin-top: 20px;
    }
    .sub-header {
        font-size: 24px;
        color: #117A65;
        margin-top: 20px;
    }
    .content-text {
        font-size: 18px;
        color: #424949;
    }
    .image-container {
        display: flex;
        justify-content: center;
        margin-top: 20px;
    }
    .button-container {
        display: flex;
        justify-content: center;
        margin-top: 20px;
    }
    .predict-button {
        background-color: #3498DB;
        color: white;
        border: none;
        padding: 10px 20px;
        font-size: 18px;
        border-radius: 10px;
        margin-top: 20px;
    }
    .success-message {
        color: #28A745;
        font-size: 24px;
        font-weight: bold;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About Project", "Prediction"])

# Main Page
if app_mode == "Home":
    st.markdown('<div class="main-header">FRUITS & VEGETABLES RECOGNITION SYSTEM</div>', unsafe_allow_html=True)
    image_path = "home_img.jpeg"
    st.image(image_path, use_column_width=True)

# About Project
elif app_mode == "About Project":
    st.markdown('<div class="main-header">About Project</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">About Dataset</div>', unsafe_allow_html=True)
    st.markdown('<div class="content-text">This dataset contains images of the following food items:</div>', unsafe_allow_html=True)
    st.code("Fruits: banana, apple, pear, grapes, orange, kiwi, watermelon, pomegranate, pineapple, mango.")
    st.code("Vegetables: cucumber, carrot, capsicum, onion, potato, lemon, tomato, radish, beetroot, cabbage, lettuce, spinach, soy bean, cauliflower, bell pepper, chili pepper, turnip, corn, sweetcorn, sweet potato, paprika, jalapeño, ginger, garlic, peas, eggplant.")
    st.markdown('<div class="sub-header">Content</div>', unsafe_allow_html=True)
    st.markdown('<div class="content-text">This dataset contains three folders:</div>', unsafe_allow_html=True)
    st.markdown('<div class="content-text">1. Train (100 images each)</div>', unsafe_allow_html=True)
    st.markdown('<div class="content-text">2. Test (10 images each)</div>', unsafe_allow_html=True)
    st.markdown('<div class="content-text">3. Validation (10 images each)</div>', unsafe_allow_html=True)

# Prediction Page
elif app_mode == "Prediction":
    st.markdown('<div class="main-header">Model Prediction</div>', unsafe_allow_html=True)
    test_image = st.file_uploader("Choose an Image:", type=["jpg", "jpeg", "png"])
    if test_image:
        st.image(test_image, use_column_width=True, caption="Uploaded Image")
    
    if st.button("Predict", key='predict', help='Click to predict the image'):
        st.snow()
        st.markdown('<div class="content-text">Our Prediction:</div>', unsafe_allow_html=True)
        result_index = model_prediction(test_image)
        # Reading Labels
        with open("labels.txt") as f:
            content = f.readlines()
        label = [i.strip() for i in content]
        st.markdown(f'<div class="success-message">Model is predicting that the image is: {label[result_index]}</div>', unsafe_allow_html=True)
