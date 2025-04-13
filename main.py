import streamlit as st
import tensorflow as tf
import numpy as np

# Embedded CSS as a multi-line string
css_styles = """
<style>
button {
    background-color: #4CAF50;
    color: white;
    padding: 10px 20px;
    border: none;
    border-radius: 5px;
    cursor: pointer;
    font-weight: bold;
}

button:hover {
    background-color: #45a049;
}

img {
    border-radius: 10px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}

div[data-testid="stSidebar"] {
    background-color: #F5F5F6;
    color: #212121;
    padding: 20px;
}

.main {
    padding: 20px;
}

@import url('https://fonts.googleapis.com/css2?family=Roboto&display=swap');
.main {
    font-family: 'Roboto', sans-serif;
}
</style>
"""

# Inject CSS styles into the app
st.markdown(css_styles, unsafe_allow_html=True)

def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_model.h5")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(64, 64))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    predictions = model.predict(input_arr)
    return np.argmax(predictions)

st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Select Page", ["🏠 Home", "📄 About Project", "🔍 Prediction"])

if app_mode == "🏠 Home":
    st.title("🍎🥕 FRUITS & VEGETABLES RECOGNITION SYSTEM 🥦🍇")
    st.image("Fruits.jpg", use_column_width=True)
    st.markdown("""
    ## Welcome!
    
    Our innovative system leverages the power of **Artificial Intelligence (AI)** and **Machine Learning (ML)** to accurately identify different fruits and vegetables through image recognition using a **Convolutional Neural Network (CNN)**.
    
    ### Key Features:
    
    - **High Accuracy Recognition:**  
      Our CNN model, built with TensorFlow and Keras, is trained on a comprehensive dataset encompassing a wide variety of fruits and vegetables. This ensures the highest possible accuracy under various lighting and environmental conditions.
    
    - **User-Friendly Interface:**  
      Easily upload an image of your fruit or vegetable and get real-time predictions. Our streamlined and intuitive user interface makes it simple for anyone to use, regardless of technical background.
    
    - **Educational Insights:**  
      Not only do you get an accurate prediction, but you also learn more about the food item. Explore details about nutritional benefits, seasonal availability, and culinary uses.
    
    - **Innovative Deep Learning Technology:**  
      The system employs advanced techniques in deep learning, ensuring that the neural network adapts to variations in image quality and composition, providing robust performance across diverse scenarios.
    
    ### How It Works:
    
    1. **Data Collection & Processing:**  
       A large dataset is used, containing images categorized into various fruits and vegetables. The data is split into training, testing, and validation sets to ensure balanced learning.
    
    2. **Training the CNN:**  
       Using TensorFlow, our CNN model learns from these images, recognizing patterns and features that are unique to each fruit and vegetable.
    
    3. **Real-Time Prediction:**  
       Once the user uploads an image, the system processes it and predicts the most likely category for the depicted fruit or vegetable.
    
    Dive into the world of AI and ML with us, and experience state-of-the-art image recognition technology firsthand!
    """)

elif app_mode == "📄 About Project":
    st.header("About Project")
    
    st.subheader("Project Overview")
    st.markdown("""
    This project leverages advanced Artificial Intelligence and Deep Learning techniques to build a robust fruits and vegetables recognition system. Through the use of a **Convolutional Neural Network (CNN)** built with TensorFlow and Keras, the system is capable of accurately identifying various fruits and vegetables from images. The goal is to simplify the classification process and provide users with an engaging, informative experience that combines technology with everyday life.
    """)
    
    st.subheader("About Dataset")
    st.markdown("""
    The foundation of this project is built on a comprehensive dataset curated specifically for image recognition tasks. The dataset includes high-quality images of a wide variety of fruits and vegetables, captured under different lighting conditions and from various angles.
    
    **Dataset Categories:**
    - **Fruits:** banana, apple, pear, grapes, orange, kiwi, watermelon, pomegranate, pineapple, mango.
    - **Vegetables:** cucumber, carrot, capsicum, onion, potato, lemon, tomato, radish, beetroot, cabbage, lettuce, spinach, soy bean, cauliflower, bell pepper, chilli pepper, turnip, corn, sweetcorn, sweet potato, paprika, jalapeño, ginger, garlic, peas, eggplant.
    """)

    st.subheader("Dataset Structure & Content")
    st.markdown("""
    The dataset is organized into three main folders:
    
    1. **Train:** Contains 100 images per category. These images are used to teach the CNN model to identify and learn unique features of each fruit and vegetable.
    2. **Test:** Consists of 10 images per category. This set is used to evaluate the model's performance and ensure that it generalizes well to unseen data.
    3. **Validation:** Also contains 10 images per category. The validation dataset helps in fine-tuning the model during training, monitoring performance, and preventing overfitting.
    """)

    st.subheader("Data Preprocessing and Augmentation")
    st.markdown("""
    To enhance the model's performance and robustness, the following data preprocessing techniques are applied:
    
    - **Resizing:** All images are resized to 64x64 pixels to maintain uniformity.
    - **Normalization:** Pixel values are scaled to ensure that the data is consistent for the CNN.
    - **Augmentation:** Techniques such as rotation, flipping, and zooming are applied to augment the dataset, increasing diversity and reducing the possibility of overfitting.
    """)

    st.subheader("Model Training & Evaluation")
    st.markdown("""
    The Convolutional Neural Network (CNN) is designed and trained with the following steps:
    
    - **Architecture Design:** The CNN architecture consists of multiple convolutional layers followed by pooling and fully connected layers, optimized for image classification tasks.
    - **Training:** The model is trained using the training dataset, employing techniques like dropout and batch normalization to improve accuracy.
    - **Evaluation:** The model's performance is evaluated on the test dataset, measuring accuracy, precision, recall, and loss metrics. Fine-tuning is performed using the validation set.
    
    Through iterative experimentation and hyperparameter tuning, the final model achieves high precision in classifying fruits and vegetables.
    """)

    st.subheader("Future Enhancements")
    st.markdown("""
    While the current model delivers robust performance, there are several opportunities for future enhancements:
    
    - **Dataset Expansion:** Incorporating more diverse images and additional categories.
    - **Advanced Architectures:** Experimenting with deeper networks or transfer learning with pre-trained models for improved accuracy.
    - **Real-Time Detection:** Integrating the model with mobile and web applications for real-time recognition.
    - **User Feedback:** Utilizing user feedback for continuous improvement through active learning.
    """)

    st.markdown("Explore the project to learn more about the technical details and the exciting potential of AI and ML in everyday applications!")


elif app_mode == "🔍 Prediction":
    st.header("Model Prediction")
    col1, col2 = st.columns(2)
    with col1:
        test_image = st.file_uploader("Choose an Image:", type=["jpg", "png", "jpeg"])
        if test_image is not None:
            st.image(test_image, use_column_width=True)
    with col2:
        if st.button("Predict"):
            if test_image is not None:
                st.snow()
                with st.spinner("Processing your image..."):
                    result_index = model_prediction(test_image)
                    with open("labels.txt") as f:
                        content = f.readlines()
                    label = [i.strip() for i in content]
                    st.success(f"Model is Predicting it's a {label[result_index]}")
            else:
                st.warning("Please upload an image first.")
