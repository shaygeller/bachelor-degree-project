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
import numpy as np

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
num_classes = 10

y_test = keras.utils.to_categorical(y_test, num_classes)

print(x_test.shape[0], 'test samples')
x_test = x_test.astype('float32')
x_test /= 255
print(x_test.shape)
loaded_model = load_model('saved_models/keras_cifar10_trained_model_32_100_201801140709.h5')


# we build a new model with the activations of the old model
# this model is truncated after the first layer
model2 = Sequential()
model2.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=x_train.shape[1:],
                  weights=loaded_model.layers[0].get_weights()))
train_activations = model2.predict(x_train)
test_activations = model2.predict(x_test)

train_filename = "train_filtered_data.txt"
test_filename = "test_filtered_data.txt"

np.save(train_filename, train_activations)
np.save(test_filename,test_activations)
