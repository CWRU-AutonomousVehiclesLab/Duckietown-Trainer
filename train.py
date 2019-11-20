from frankmodel import FrankNet
from log_reader import Reader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import shuffle
from keras.optimizers import Adam
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
matplotlib.use("Agg")


#! Training Configuration
EPOCHS = 10
INIT_LR = 1e-3
BS = 32
IMAGE_DIMS = (96, 96, 3)

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
    return


#!================================================================
load_data()
print('Load all complete')
# split data into training and validation
observation_train, observation_valid, linear_train, linear_valid, angular_train, angular_valid = train_test_split(
    observation, linear, angular, test_size=0.2)

# define the network model
model = FrankNet.build(200, 100)

losses = {
    "Linear_Velocity_Out": "mse",
    "Angular_Velocity_Out": "mse"
}
lossWeights = {"Linear_Velocity_Out": 1.0, "Angular_Velocity_Out": 1.0}

opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(optimizer=opt, loss=losses, loss_weights=lossWeights,
              metrics=["accuracy"])


history = model.fit(observation_train,
                    {"Linear_Velocity_Out": linear_train,
                        "Angular_Velocity_Out": angular_train},
                    validation_data=(observation_valid, {
                                     "Linear_Velocity_Out": linear_valid, "Angular_Velocity_Out": angular_valid}),
                    epochs=EPOCHS, verbose=1)

model.save('FrankNet.h5')
