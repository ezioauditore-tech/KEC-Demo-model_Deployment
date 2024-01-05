# train_model.py

import tensorflow as tf
from tensorflow.keras import layers, models, datasets
from sklearn.model_selection import train_test_split
from fastapi import FastAPI, HTTPException

# Load and preprocess the FashionMNIST dataset
(X_train, y_train), (X_test, y_test) = datasets.fashion_mnist.load_data()
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)).astype('float32') / 255.0
X_val = X_val.reshape((X_val.shape[0], 28, 28, 1)).astype('float32') / 255.0
X_test = X_test.reshape((X_test.shape[0], 28, 28, 1)).astype('float32') / 255.0

# Define the neural network model
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28, 1)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=5, validation_data=(X_val, y_val))

# Save the model
model.save("fashion_mnist_model")

# Run this script to train the model and save it
