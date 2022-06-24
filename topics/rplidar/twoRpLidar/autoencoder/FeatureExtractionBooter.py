import os
import pickle

import numpy as np
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense

from MachineSettings import MachineSettings
from ctumrs.topics.rplidar.twoRpLidar.autoencoder.Plots import Plots
from mMath.data.preProcess.RowsNormalizer import RowsNormalizer


class FeatureExtractionBooter:
    @staticmethod
    def extractFeatures()->None:
        #settings
        scenarioName = "normal-scenario"

        #leader or follower
        leadership = "leader"

        rowsNum = 35000

        #some websites say epochs must start from three times the number of the columns
        epochs = 1000

        #How many data per time feed into the NN for training
        #some websites siad that this amount is the best
        batchSize = 32

        # This is the dimension of the original space
        inputDim = 720

        # This is the dimension of the latent space (encoding space)
        latentDim = 3

        sharedDataPathToLidarsScenario = MachineSettings.MAIN_PATH+"projs/research/data/self-aware-drones/ctumrs/two-drones/{}/lidars/".format(scenarioName)

        #Loading data
        twoLidarsTimeRangesObssPickleFile = open(sharedDataPathToLidarsScenario+"twoLidarsTimeRangesObss.pkl", 'rb')
        pklDict = pickle.load(twoLidarsTimeRangesObssPickleFile)
        npLeaderRangesObss = np.array(pklDict["{}TimeRangesObss".format(leadership)])[:rowsNum, 1:]
        normalizedNpLeaderRangesObss = RowsNormalizer.getNpNormalizedNpRows(npLeaderRangesObss)



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


        modelHistory = autoencoder.fit(normalizedNpLeaderRangesObss
                                       , normalizedNpLeaderRangesObss
                                       , epochs=epochs
                                       , batch_size=batchSize
                                       , verbose=0)

        #saving the model
        autoencoder.save(filepath = sharedDataPathToLidarsScenario+"autoencoders/{}-encoder-decoder-rows-num-{}-epochs-{}-batch-size-{}.h5".format(leadership,rowsNum,epochs,batchSize))
        decoder.save(filepath = sharedDataPathToLidarsScenario+"autoencoders/{}-decoder-rows-num-{}-epochs-{}-batch-size-{}.h5".format(leadership,rowsNum,epochs,batchSize))
        encoder.save(filepath = sharedDataPathToLidarsScenario+"autoencoders/{}-encoder-rows-num-{}-epochs-{}-batch-size-{}.h5".format(leadership,rowsNum,epochs,batchSize))

        #plot Loss vs Epoch
        Plots.plotLossVsEpoch(modelHistory.history["loss"])


        #let check the latent space
        encodedXtrain = encoder(normalizedNpLeaderRangesObss)

        # Plots.plot2DEncodedXTrain(encodedXtrain)
        Plots.plot3DEncodedXTrain(encodedXtrain)

if __name__=="__main__":
    FeatureExtractionBooter.extractFeatures()
    os.system('spd-say "Feature extraction finished"')