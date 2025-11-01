import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ✅ Correct path – data folder is inside scripts
data_dir = os.path.join(os.path.dirname(__file__), 'data')

categories = os.listdir(data_dir)
img_size = 128

data = []
labels = []

# ✅ Load and preprocess images
for idx, category in enumerate(categories):
    folder = os.path.join(data_dir, category)
    for img in os.listdir(folder):
        img_path = os.path.join(folder, img)
        try:
            img_arr = cv2.imread(img_path)
            img_arr = cv2.resize(img_arr, (img_size, img_size))
            data.append(img_arr)
            labels.append(idx)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")

# Convert to numpy arrays
data = np.array(data) / 255.0
labels = to_categorical(np.array(labels))

# Split into training and testing data
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)

# ✅ Build model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(img_size, img_size, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(len(categories), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ✅ Train model
print("🚀 Training model, please wait...")
history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# ✅ Save trained model
model.save('rps_model.h5')
print("✅ Model saved as rps_model.h5")

# ✅ Optional: visualize training progress
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend()
plt.title('Accuracy')

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title('Loss')
plt.show()


