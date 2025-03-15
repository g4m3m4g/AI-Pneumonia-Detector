from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
import config
import Preprocess

img_size = config.img_size 

# Load ResNet50 (pre-trained)
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))

# Freeze base model layers
for layer in base_model.layers:
    layer.trainable = False

# Define the custom model
model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary classification (Pneumonia vs. Normal)
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(Preprocess.train_generator, validation_data=Preprocess.val_generator, epochs=10)

# Save the trained model
model.save("pneumonia_model.h5")
print("Model saved as pneumonia_model.h5")
