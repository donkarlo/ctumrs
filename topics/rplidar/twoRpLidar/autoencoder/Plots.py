import pickle
from builtins import range
from turtle import color

import numpy as np
from matplotlib import pyplot as plt
import random
from tensorflow.keras.models import Model, Sequential,load_model
from tensorflow.keras.layers import Dense

from MachineSettings import MachineSettings
from mMath.data.preProcess.RowsNormalizer import RowsNormalizer


class Plots:
    @staticmethod
    def plotOriginalVsReconstructed(title, xTrain, colNames,autoencoder):
        fig = plt.figure(figsize=(10,6))
        plt.suptitle(title)
        for i in range(3):
            #give me three plots in three and one column, this plot is the i+1th plot
            plt.subplot(3, 1, i+1)
            #choose one random index from data rows
            randomDataRowNum = random.sample(range(0,xTrain.shape[0]), 1)
            #Squeeze, remove axis of length one
            plt.plot(autoencoder.predict(xTrain[randomDataRowNum]).squeeze()
                     , label='reconstructed' if i == 0 else '')
            plt.plot(xTrain[randomDataRowNum].squeeze(),
                     label='original' if i == 0 else '')
            fig.axes[i].set_xticklabels(colNames)
            #10 is num of ticks which is equal to num of colNames
            # gives an array from 0 to 9
            plt.xticks(np.arange(0, 10, 1))
            plt.grid(True)
            if i == 0:
                plt.legend()

    @staticmethod
    def plotLossVsEpoch(modelHistoryLoss):
        plt.plot(modelHistoryLoss)
        plt.title("Loss vs. Epoch")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.grid(True)
        plt.show()

    @staticmethod
    def plot2DEncodedXTrain(encodedXtrain):
        plt.figure(figsize=(6, 6))
        #encodedXtrain[:, 0] from the begining to the end, put the first component in an array
        #encodedXtrain[:, 1] from the begining to the end, put the second component in an array
        plt.scatter(encodedXtrain[:, 0], encodedXtrain[:, 1],alpha=0.4)
        plt.xlabel('Latent Dimension 1')
        plt.ylabel('Latent Dimension 2')
        plt.show()

    @staticmethod
    def plot3DEncodedXTrain(encodedXtrain):
        plt.figure(figsize=(6, 6))
        ax = plt.axes(projection="3d")

        # Creating plot
        ax.scatter3D(encodedXtrain[:, 0]
                     , encodedXtrain[:, 1]
                     , encodedXtrain[:, 2]
                     , s = 0.05
                     )
        plt.title("Latent dimentions")

        # show plot
        plt.show()


if __name__=="__main__":
    ############ settings
    scenarioName = "normal-scenario"

    # leader or follower
    leadership = "leader"

    rowsNum = 50000

    # some websites say epochs must start from three times the number of the columns
    epochs = 2160

    # How many data per time feed into the NN for training
    # some websites siad that this amount is the best
    batchSize = 32

    # This is the dimension of the original space
    inputDim = 720

    # This is the dimension of the latent space (encoding space)
    latentDim = 3

    sharedDataPathToLidarsScenario = MachineSettings.MAIN_PATH + "projs/research/data/self-aware-drones/ctumrs/two-drones/{}/lidars/".format(
        scenarioName)

    # Loading data
    twoLidarsTimeRangesObssPickleFile = open(sharedDataPathToLidarsScenario + "twoLidarsTimeRangesObss.pkl", 'rb')
    pklDict = pickle.load(twoLidarsTimeRangesObssPickleFile)
    npLeaderRangesObss = np.array(pklDict["{}TimeRangesObss".format(leadership)])[:rowsNum, 1:]
    normalizedNpLeaderRangesObss = RowsNormalizer.getNpNormalizedNpRows(npLeaderRangesObss)

    #load model
    encoder = load_model(filepath = sharedDataPathToLidarsScenario+"autoencoders/{}-encoder-rows-num-{}-epochs-{}-batch-size-{}.h5".format(leadership,rowsNum,epochs,batchSize))
    # let check the latent space
    encodedXtrain = encoder(normalizedNpLeaderRangesObss)

    Plots.plot3DEncodedXTrain(encodedXtrain)

