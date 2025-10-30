import gradio as gr
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

model = tf.keras.models.load_model("plant_disease_model.h5")

class_names = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "PlantVillage"
]

def predict_plant(img):
    img = img.resize((64, 64))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    confidence = round(100 * np.max(prediction), 2)
    return {class_names[class_index]: confidence}

interface = gr.Interface(
    fn=predict_plant,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=1),
    title="🌿 Plant Disease Classifier",
    description="Upload an image of a plant leaf to predict its disease condition."
)

interface.launch()
