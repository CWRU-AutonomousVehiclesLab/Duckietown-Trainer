from frankmodel import FrankNet
from log_reader import Reader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import shuffle
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.utils import multi_gpu_model
from keras.utils import plot_model
from keras.callbacks import TensorBoard
from time import time

import numpy as np
import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt
import os
import csv
import cv2
import math
import random
import matplotlib
import tkinter

matplotlib.use('TkAgg')


#! Training Configuration
EPOCHS = 10000
INIT_LR = 1e-3
BS = 64
GPU_COUNT = 3

#! Log Interpretation
STORAGE_LOCATION = "trained_models/behavioral_cloning"

#! Global
observation = []
linear = []
angular = []


def load_data():
    global observation, linear, angular
    reader = Reader('train.log')
    observation, linear, angular = reader.read()
    observation = np.array(observation)
    linear = np.array(linear)
    angular = np.array(angular)
    print('Observation Length: ',len(observation))
    print('Linear Length: ',len(linear))
    print('Angular Length: ',len(angular))
    #exit()
    return


#-----------------------------------------------------------------------------
# Define custom loss functions for regression in Keras 
#-----------------------------------------------------------------------------

# root mean squared error (rmse) for regression
def rmse(y_true, y_pred):
    from keras import backend
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))

# mean squared error (mse) for regression
def mse(y_true, y_pred):
    from keras import backend
    return backend.mean(backend.square(y_pred - y_true), axis=-1)

# coefficient of determination (R^2) for regression
def r_square(y_true, y_pred):
    from keras import backend as K
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return (1 - SS_res/(SS_tot + K.epsilon()))

def r_square_loss(y_true, y_pred):
    from keras import backend as K
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return 1 - ( 1 - SS_res/(SS_tot + K.epsilon()))

#!================================================================
load_data()
print('Load all complete')

# define the network model
single_model = FrankNet.build(200, 100)

losses = {
    "Linear": "mse",
    "Angular": "mse"
}
lossWeights = {"Linear": 0.3, "Angular": 1.0}


opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)

metrics_list = ["mse", rmse, r_square]

model = multi_gpu_model(single_model, gpus=GPU_COUNT)
#model = single_model

model.compile(optimizer=opt, loss=losses, loss_weights=lossWeights,
              metrics=metrics_list)

plot_model(model, to_file='model.png')

# tensorboard
tensorboard = TensorBoard(log_dir='logs/{}'.format(time()))

# checkpoint
filepath="FrankNetBest.h5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint,tensorboard]

history = model.fit(observation,
                    {"Linear": linear,
                        "Angular": angular},
                    epochs=EPOCHS,callbacks=callbacks_list, verbose=1)

model.save('FrankNet.h5')

# list all data in history
print(history.history.keys())
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for accuracy
plt.plot(history.history['Linear_accuracy'])
plt.plot(history.history['Angular_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Lienar', 'Angular'], loc='upper left')
plt.show()
