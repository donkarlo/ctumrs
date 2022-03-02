import numpy as np
from matplotlib import pyplot as plt
import random

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


def plotLossVsEpoch(modelHistoryLoss):
    plt.plot(modelHistoryLoss)
    plt.title("Loss vs. Epoch")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.grid(True)
    plt.show()

def plot2DEncodedXTrain(encodedXtrain):
    plt.figure(figsize=(6, 6))
    #encodedXtrain[:, 0] from the begining to the end, put the first component in an array
    #encodedXtrain[:, 1] from the begining to the end, put the second component in an array
    plt.scatter(encodedXtrain[:, 0], encodedXtrain[:, 1],alpha=0.4)
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.show()

    # ax = plt.axes(projection="3d")
    #
    # # Creating plot
    # ax.scatter3D(encodedXtrain[:, 0], encodedXtrain[:, 1], encodedXtrain[:, 2], color="green")
    # plt.title("Latent dimentions")
    #
    # # show plot
    # plt.show()

def plot3DEncodedXTrain(encodedXtrain):
    plt.figure(figsize=(6, 6))
    ax = plt.axes(projection="3d")

    # Creating plot
    ax.scatter3D(encodedXtrain[:, 0], encodedXtrain[:, 1], encodedXtrain[:, 2],alpha=0.4)
    plt.title("Latent dimentions")

    # show plot
    plt.show()
