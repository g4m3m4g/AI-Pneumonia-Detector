import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import config

# Dataset path
data_dir = "chest_xray"
img_size = config.img_size 

# Data Augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255, rotation_range=20, width_shift_range=0.2,
    height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    os.path.join(data_dir, "train"), target_size=(img_size, img_size),
    batch_size=32, class_mode='binary'
)

val_generator = val_datagen.flow_from_directory(
    os.path.join(data_dir, "val"), target_size=(img_size, img_size),
    batch_size=32, class_mode='binary'
)
