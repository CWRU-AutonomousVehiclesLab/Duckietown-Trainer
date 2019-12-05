from frankmodel import FrankNet
from log_reader import Reader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import shuffle
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.utils import multi_gpu_model

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
BS = 19000
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


#!================================================================
load_data()
print('Load all complete')

# define the network model
single_model = FrankNet.build(200, 100)

losses = {
    "Linear_Velocity_Out": "mse",
    "Angular_Velocity_Out": "mse"
}
lossWeights = {"Linear_Velocity_Out": 1.0, "Angular_Velocity_Out": 1.0}

opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model = multi_gpu_model(single_model, gpus=GPU_COUNT)

model.compile(optimizer=opt, loss=losses, loss_weights=lossWeights,
              metrics=["accuracy"])

# checkpoint
filepath="FrankNetBest.h5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

history = model.fit(observation,
                    {"Linear_Velocity_Out": linear,
                        "Angular_Velocity_Out": angular},
                    epochs=EPOCHS,callbacks=callbacks_list, verbose=1)

model.save('FrankNet.h5')
"""
dict_keys([ 'val_loss', 
            'val_Linear_Velocity_Out_loss', 
            'val_Angular_Velocity_Out_loss', 
            'val_Linear_Velocity_Out_accuracy', 
            'val_Angular_Velocity_Out_accuracy', 
            'loss', 
            'Linear_Velocity_Out_loss', 
            'Angular_Velocity_Out_loss', 
            'Linear_Velocity_Out_accuracy', 
            'Angular_Velocity_Out_accuracy'])

"""
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
plt.plot(history.history['Linear_Velocity_Out_accuracy'])
plt.plot(history.history['Angular_Velocity_Out_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Lienar', 'Angular'], loc='upper left')
plt.show()
