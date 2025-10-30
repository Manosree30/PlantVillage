import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam


base_dir = r"C:\Users\manos\OneDrive\Desktop\python program\CNN"
dataset_dir = os.path.join(base_dir, "archive", "PlantVillage", "PlantVillage")

train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")


if not os.path.exists(train_dir) or not os.path.exists(test_dir):
    print("📂 Splitting dataset into train and test folders...")
    for class_name in os.listdir(dataset_dir):
        class_path = os.path.join(dataset_dir, class_name)
        if os.path.isdir(class_path):
            images = os.listdir(class_path)
            train_imgs, test_imgs = train_test_split(images, test_size=0.2, random_state=42)

            os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
            os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

            for img in train_imgs:
                shutil.copy(os.path.join(class_path, img), os.path.join(train_dir, class_name, img))
            for img in test_imgs:
                shutil.copy(os.path.join(class_path, img), os.path.join(test_dir, class_name, img))

    print("✅ Dataset successfully split into Train and Test folders!")
else:
    print("✅ Train/Test folders already exist. Skipping split...")


train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_set = train_datagen.flow_from_directory(
    train_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical'
)

test_set = test_datagen.flow_from_directory(
    test_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical'
)


model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),

    Dense(train_set.num_classes, activation='softmax')
])

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()


history = model.fit(
    train_set,
    epochs=10,
    validation_data=test_set
)


test_loss, test_acc = model.evaluate(test_set)
print(f"\n✅ Test Accuracy: {test_acc*100:.2f}%")
print(f"🧮 Test Loss: {test_loss:.4f}")


plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend(); plt.title("Accuracy")

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend(); plt.title("Loss")

plt.tight_layout()
plt.savefig("training_results.png")
print("📊 Training graph saved as training_results.png")


model.save("plant_disease_model.h5")
print("💾 Model saved as 'plant_disease_model.h5'")
