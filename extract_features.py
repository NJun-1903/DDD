import tensorflow as tf
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import cv2
import os


def extract_features(image_path, base_model):
    """Trích xuất đặc trưng cho một ảnh đơn lẻ"""
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image from path: {image_path}")

        image = cv2.resize(image, (224, 224))
        img_array = img_to_array(image)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Trích xuất đặc trưng sử dụng GPU
        with tf.device('/GPU:0'):
            feature_map = base_model.predict(img_array, verbose=0)
            avg_pooled_features = tf.reduce_mean(feature_map, axis=(1, 2))

        return avg_pooled_features.numpy()[0]
    except Exception as e:
        print(f"Error extracting features from {image_path}: {e}")
        return None
    
