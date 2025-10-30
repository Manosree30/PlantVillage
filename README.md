# 🌿 Plant Disease Detection using CNN and Gradio

## 📘 Project Overview
This project uses **Convolutional Neural Networks (CNN)** to detect **plant leaf diseases** from images.  
It was trained on the **PlantVillage dataset**, which contains healthy and infected leaves of various plants such as **Apple, Tomato, and Pepper**.  
A **Gradio web interface** is built to easily upload an image and view the prediction in real-time.

---

## 🎯 Aim
To develop a deep learning model using CNN that can accurately identify plant diseases from leaf images and deploy it through an interactive Gradio web app.

---

- The dataset was split into **80% training** and **20% testing** using a Python script.

### 2. Model Building
- Used **TensorFlow and Keras** to build the CNN.
- Applied **ImageDataGenerator** for image augmentation.
- Model layers included:
- Convolutional layers (feature extraction)
- MaxPooling layers (dimensionality reduction)
- Flatten and Dense layers (classification)

## Model Training

Trained the model for multiple epochs.
Achieved ≈ 94% test accuracy.

## Deployment using Gradio

A Gradio interface was created (app_gradio.py) to upload images and predict disease.
### Command to run the app:
python app_gradio.py
### The app runs locally at:
http://127.0.0.1:7860

## Libraries Used
TensorFlow / Keras
NumPy
Matplotlib
Gradio
scikit-learn
os, shutil (for dataset organization)
### Install all dependencies with:
pip install tensorflow gradio numpy matplotlib scikit-learn

## Results
| Metric   | Training | Testing |
| -------- | -------- | ------- |
| Accuracy | ~98%     | ~94%    |
| Loss     | ↓        | 0.1437  |

 Final Test Accuracy: 93.75%
 Training graph: training_results.png
## Result / Conclusion

The CNN model successfully classifies plant leaf diseases with high accuracy.
This project demonstrates how deep learning + Gradio can help in real-world agricultural disease diagnosis.
