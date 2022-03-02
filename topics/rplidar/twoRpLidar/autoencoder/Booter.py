import os
import pickle

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense

from sklearn.preprocessing import StandardScaler
from ctumrs.topics.rplidar.twoRpLidar.autoencoder.Plots import plotOriginalVsReconstructed, plotLossVsEpoch, \
    plot2DEncodedXTrain,plot3DEncodedXTrain





def getNpNormalizedHighDimData(xTrain:np.ndarray):


    # Scale data to have zero mean and unit variance
    scaler = StandardScaler()
    scaler.fit(xTrain)
    xTrain = scaler.transform(xTrain)

    return xTrain


pickleFile = open("/home/donkarlo/Dropbox/projs/research/data/self-aware-drones/ctumrs/two-drones/normal-scenario/lidars/twoLidarsTimeRangesObss.pkl", 'rb')
pklDict = pickle.load(pickleFile)
leaderTimeTimeRangesObssList = np.array(pklDict["leaderTimeRangesObss"])[:30000,1:]



xTrain = getNpNormalizedHighDimData(leaderTimeTimeRangesObssList)

# This is the dimension of the original space
inputDim = 720

# This is the dimension of the latent space (encoding space)
latentDim = 2

encoder = Sequential([
    Dense(512, activation='relu', input_shape=(inputDim,)),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(latentDim, activation='relu')
])

decoder = Sequential([
    Dense(32, activation='relu', input_shape=(latentDim,)),
    Dense(64, activation='relu'),
    Dense(128, activation='relu'),
    Dense(256, activation='relu'),
    Dense(512, activation='relu'),
    Dense(inputDim, activation=None)
])

autoencoder = Model(inputs=encoder.input, outputs=decoder(encoder.output))
autoencoder.compile(loss='mse', optimizer='adam')

modelHistory = autoencoder.fit(xTrain, xTrain, epochs=100, batch_size=10, verbose=0)
#plot Loss vs Epoch
plotLossVsEpoch(modelHistory.history["loss"])


#let check the latent space
encodedXtrain = encoder(xTrain)
plot2DEncodedXTrain(encodedXtrain)
# plot3DEncodedXTrain(encodedXtrain)

os.system('spd-say "your program has finished"')
