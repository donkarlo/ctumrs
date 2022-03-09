import os
import pickle

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense

from ctumrs.topics.rplidar.twoRpLidar.autoencoder.Plots import Plots
from mMath.data.preProcess.PreProcess import PreProcess


class FeatureExtractionBooter:
    @staticmethod
    def extractFeatures()->None:
        #settings
        scenarioName = "normal-scenario"
        leaderFollower = "follower"
        sharedDataPathToLidarsScenario = "/home/donkarlo/Dropbox/projs/research/data/self-aware-drones/ctumrs/two-drones/{}/lidars/".format(scenarioName)

        #Loading data
        twoLidarsTimeRangesObssPickleFile = open(sharedDataPathToLidarsScenario+"twoLidarsTimeRangesObss.pkl", 'rb')
        pklDict = pickle.load(twoLidarsTimeRangesObssPickleFile)
        npLeaderRangesObss = np.array(pklDict["{}TimeRangesObss".format(leaderFollower)])[:50000, 1:]



        normalizedNpLeaderRangesObss = PreProcess.getNpNormalizedNpData(npLeaderRangesObss)

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


        modelHistory = autoencoder.fit(normalizedNpLeaderRangesObss, normalizedNpLeaderRangesObss, epochs=50, batch_size=10, verbose=0)


        encoder.save(filepath = sharedDataPathToLidarsScenario+"autoencoders/{}-encoder.h5".format(leaderFollower))
        #plot Loss vs Epoch
        Plots.plotLossVsEpoch(modelHistory.history["loss"])


        #let check the latent space
        encodedXtrain = encoder(normalizedNpLeaderRangesObss)
        Plots.plot2DEncodedXTrain(encodedXtrain)
        # Plots.plot3DEncodedXTrain(encodedXtrain)

if __name__=="__main__":
    FeatureExtractionBooter.extractFeatures()
    os.system('spd-say "Feature extraction finished"')