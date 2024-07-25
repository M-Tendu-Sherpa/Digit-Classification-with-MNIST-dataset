#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


import ssl
## Disabling the SSL temporarily due the SSL certificate error while downloading the MNIST dataset. 
## However, it is not recommended.
ssl._create_default_https_context = ssl._create_unverified_context


(x_train, y_train), (x_test, y_test) = mnist.load_data()


## Enabling the SSL again just in case we proceed with any more downloading from other sources.
ssl._create_default_https_context = ssl.create_default_context


## Normalizing the datasets ranging from 0 to 1 for faster convergence
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)


## Reshaping the data by adding channel value for CNN.
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))


## Converting the ground truth labels to categorical values.
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)


## Creating a sequential model
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])


## Compiling the model.
model.compile(optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'])


## Training the model with 5 epochs
model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.2)


## Testing the model for accuracy and loss.
test_loss, test_acc = model.evaluate(x_test, y_test)


print(f'Test accuracy: {test_acc:.4f} /n Test loss: {test_loss:.4f}')


import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


## Importing the images with numbers from local folder and testing the model.
for x in range(0,10):
    img = cv.imread(f'./Digits/{x}.png')[:,:,1]
    img = np.invert(np.array([img]))
    prediction = model.predict(img)
    print(f'Predicted label: {np.argmax(prediction)}')
    plt.imshow(img[0], cmap=plt.cm.binary)
    plt.show()


