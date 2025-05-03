Fruits & Vegetables Recognition System

An AI-powered application that leverages Convolutional Neural Networks (CNN) to accurately classify images of fruits and vegetables. This system is designed to assist in automating the identification process, which can be beneficial for various applications such as inventory management, dietary tracking, and educational tools.


Features

Accurate Classification: Utilizes a trained CNN model to distinguish between various fruits and vegetables.

User-Friendly Interface: Simple command-line interface for image input and result display.

Extensible Architecture: Modular codebase allowing easy updates and integration with other systems.

Efficient Performance: Optimized for quick image processing and prediction.

Technologies Used

Python 3.x: Primary programming language.

TensorFlow & Keras: For building and training the CNN model.

NumPy: Handling numerical operations.

OpenCV: Image processing and manipulation.

Pillow (PIL): Image loading and preprocessing.


Fruits.jpg: Sample image used in the project.

labels.txt: Text file containing the labels corresponding to the classes the model can predict.

main.py: Main script to run the prediction.

requirement.txt / requirements.txt: List of dependencies required to run the project.

trained_model.h5: Pre-trained CNN model for fruit and vegetable classification.


⚙️ Installation & Setup

Clone the Repository:

git clone https://github.com/AashuBaria03/Fruits-Vegetables-Recognition-System.git
cd Fruits-Vegetables-Recognition-System
Create a Virtual Environment (Optional but recommended):


python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install Dependencies:


pip install -r requirements.txt


Usage
Prepare Your Image:

Place the image you want to classify in the project directory or note its path. Ensure the image is clear and contains a single fruit or vegetable.

Run the Prediction Script:


python main.py
The script will prompt you to enter the path of the image you wish to classify.

View Results:

The model will process the image and output the predicted class (e.g., "Apple", "Carrot") along with the confidence score.

📊 Model Performance

Model Architecture: Convolutional Neural Network (CNN) with multiple layers optimized for image classification tasks.

Training Data: Dataset comprising various images of fruits and vegetables, preprocessed for optimal training.

Accuracy: Achieved high accuracy on validation and test datasets, ensuring reliable predictions.

📌 Future Enhancements

GUI Integration: Develop a graphical user interface for easier interaction.

Mobile Application: Extend functionality to Android/iOS platforms.

Expanded Dataset: Include more classes and diverse images to improve model robustness.

Nutritional Information: Provide details like calories, vitamins, and minerals for each classified item.
