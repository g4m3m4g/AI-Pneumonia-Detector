import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import config

# Load trained model
model = tf.keras.models.load_model("pneumonia_model.h5")

# Dataset path
data_dir = "chest_xray"
img_size = config.img_size
batch_size = config.batch_size

# Load test data
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    os.path.join(data_dir, "test"), target_size=(img_size, img_size),
    batch_size=batch_size, class_mode='binary', shuffle=False
)

# Evaluate the model
loss, accuracy = model.evaluate(test_generator)
print(f"Test Accuracy: {accuracy:.2f}")
