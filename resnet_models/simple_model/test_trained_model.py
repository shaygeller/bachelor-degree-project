from __future__ import print_function
import datetime
import keras
from keras.callbacks import EarlyStopping
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os
import tensorflow as tf


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
num_classes = 10

y_test = keras.utils.to_categorical(y_test, num_classes)

print(x_test.shape[0], 'test samples')
x_test = x_test.astype('float32')
x_test /= 255
print(x_test.shape)
loaded_model = load_model('saved_models/keras_cifar10_trained_model_32_100_201801140709.h5')
# Score trained model.
print(loaded_model.model)
scores = loaded_model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])


