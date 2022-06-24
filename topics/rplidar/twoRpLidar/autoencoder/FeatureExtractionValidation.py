import pickle

import numpy as np
from matplotlib import pyplot as plt

from MachineSettings import MachineSettings
from mMath.data.preProcess.RowsNormalizer import RowsNormalizer
from tensorflow.keras.models import Model, Sequential,load_model
from tensorflow.keras.layers import Dense


class FeatureExtractionValidation:
    @staticmethod
    def trainTest():
        ############ settings
        scenarioName = "normal-scenario"

        # leader or follower
        leadership = "leader"

        rowsNum = 50000
        trainDataNum = 35000

        # some websites say epochs must start from three times the number of the columns
        epochs = 1000

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



        ###### Choose one data ffrom the test part
        normalizedNpTestObss = normalizedNpLeaderRangesObss[trainDataNum:]
        autoencoder = load_model(filepath=sharedDataPathToLidarsScenario + "autoencoders/{}-encoder-decoder-rows-num-{}-epochs-{}-batch-size-{}.h5".format(
                leadership, trainDataNum, epochs, batchSize))

        dists = []
        predictedNormalizedNpTestObss = autoencoder.predict(normalizedNpTestObss)
        for counter,normalizedNpTestObs in enumerate(normalizedNpTestObss):
            dists.append(np.linalg.norm(normalizedNpTestObs - predictedNormalizedNpTestObss[counter]))

        print(sum(list(map(lambda x:pow(x,2),dists)))/(rowsNum-trainDataNum))
        # plot dists
        plt.figure(figsize=(20, 6))
        plt.plot(range(0, rowsNum-trainDataNum), dists)
        plt.legend(["Real", "Reconstructed"])
        plt.show()

    @staticmethod
    def randomPoint():
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

        ########## reconstructing

        # Load auto encoder
        autoencoder = load_model(
            filepath=sharedDataPathToLidarsScenario + "autoencoders/{}-encoder-decoder-rows-num-{}-epochs-{}-batch-size-{}.h5".format(
                leadership, rowsNum, epochs, batchSize))
        # choose a random data
        realRandomNpRow = normalizedNpLeaderRangesObss[np.random.choice(normalizedNpLeaderRangesObss.shape[0]
                                                                        , 1
                                                                        , replace=False)]
        predictionForRandomRow = autoencoder.predict(realRandomNpRow)
        # plotting
        plt.figure(figsize=(20, 6))
        plt.scatter(range(0, 720), realRandomNpRow, alpha=0.2)
        plt.scatter(range(0, 720), predictionForRandomRow, color="red", alpha=0.1)
        plt.legend(["Real", "Reconstructed"])
        plt.show()


if __name__ == "__main__":
    FeatureExtractionValidation.trainTest()
    FeatureExtractionValidation.randomPoint()



