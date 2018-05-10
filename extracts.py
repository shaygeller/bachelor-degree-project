from __future__ import print_function
import datetime
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os
import tensorflow as tf
from keras.models import load_model

# Configure keras to work with GPU
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)


# Load model
saved_dir = os.path.join(os.getcwd(), "resnet_models", 'simple_model', 'saved_models')
model_name = "keras_cifar10_trained_model_32_1_201801121856.h5"
model_path = os.path.join(saved_dir, model_name)
model = load_model(model_path)
print(model)

# Get test set match to the model, and preper for evaluation by the model (normalized)
(_, _), (x_test, y_test) = cifar10.load_data()
x_test = x_test.astype('float32')
x_test /= 255
num_classes = 10
y_test = keras.utils.to_categorical(y_test, num_classes)


# # predict with model (sanity check)
# scores = model.evaluate(x_test, y_test, verbose=1)
# print('Test loss:', scores[0])
# print('Test accuracy:', scores[1])


# Get weights of the trained model
weights_list = model.get_weights()
print(len(weights_list))
for i, weights in enumerate(weights_list):
    print(weights)

# Create a new model with only one layer with the previous model's weights
model_new = Sequential()
model_new.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=x_test.shape[1:]))
model_new.add(Activation('relu'))

# for i, weights in enumerate(weights_list):
#     model_new.layers[i].set_weights(weights)

# Print and save results from the testset after the first layer
