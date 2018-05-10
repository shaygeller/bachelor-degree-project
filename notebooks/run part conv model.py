from __future__ import print_function
import datetime
import keras
from keras.callbacks import EarlyStopping
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os
import tensorflow as tf
from keras.datasets import cifar10
from keras.callbacks import CSVLogger


import numpy as np
import os

# Configure keras to work with GPU
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)



train_filename = "train_filtered_data.txt.npy"
train_filename_path = os.path.join("..","resnet_models","simple_model",train_filename)
train_after_first_layer = np.load(train_filename_path)

test_filename = "test_filtered_data.txt.npy"
test_filename_path = os.path.join("..","resnet_models","simple_model",test_filename)
test_after_first_layer = np.load(test_filename_path)



batch_size = 10
epochs = 500

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
#
for lr in [0.1,0.01]:
    print("Working on", lr, "learning rate")
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]))
    model.add(Dropout(0.2))

    # initiate RMSprop optimizer
    # lr = 0.0001
    # opt = keras.optimizers.rmsprop(lr=lr, decay=1e-6)
    opt = keras.optimizers.Adam(lr=lr)
    # opt = keras.optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)

    # Callbacks
    early_stopper = EarlyStopping(min_delta=0.001, patience=10, verbose=1)
    now_time = datetime.datetime.now()
    curr_time = now_time.strftime("%Y%m%d%H%M")
    log_name = curr_time + "_" + str(lr) + "_log.csv"
    csv_logger = CSVLogger(log_name, append=True, separator=';')

    # TODO : Play with learning rate. increase\decrease learning rate. CHECK
    # TODO : Play with learning rate. increase\decrease DROPOUT. CHECK
    # TODO : Size of error propagated in CNN (mean and std)
    # TODO : reduce train to 1000 of filtered dataset on every filter seperatly\jointly

    # Let's train the model using RMSprop
    losses = "mse"
    # mean_absolute_error
    model.compile(loss=losses,
                  optimizer=opt,
                  metrics=['accuracy'])
    model.fit(x_train, train_after_first_layer,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_test, test_after_first_layer),
            shuffle=True,
            callbacks=[early_stopper,csv_logger])


    # save_dir =
    # Save model and weights
    # if not os.path.isdir(save_dir):
        # os.makedirs(save_dir)
    # model_path = os.path.join(save_dir, model_name)
    model_name = curr_time + "_my_partial_conv_model.h5"
    model.save(model_name)
    print('Saved trained model at %s ' % model_name)
