import os
import pickle

import numpy as np

from MachineSettings import MachineSettings
from ctumrs.PosVelsClusteringStrgy import PosVelsClusteringStrgy
from ctumrs.TwoAlphabetWordsTransitionMatrix import TwoAlphabetWordsTransitionMatrix
from keras.models import load_model
from ctumrs.sensors.lidar.two.autoencoder.Plots import Plots
from mMath.calculus.derivative.TimePosRowsDerivativeComputer import TimePosRowsDerivativeComputer
from mMath.data.preProcess.RowsNormalizer import RowsNormalizer

class NormalScenarioTransitionMatrixBuilderBooter:
    @staticmethod
    def boot():
        sharedPathToTwoLidars = MachineSettings.MAIN_PATH+"projs/research/data/self-aware-drones/ctumrs/two-drones/normal-scenario/lidar/"
        pickleFileName = "twoLidarsTimeRangesObss.pkl"

        ####### Feature extraction settings
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

        ####### Transition matrix settings
        rowsNum = 50000
        velCoefficient = 10000
        leaderClustersNum = 75
        followerClustersNum = 75

        #########Load data
        pklFile = open(sharedPathToTwoLidars + pickleFileName, "rb")
        leaderFollowerTimeRangesDict = pickle.load(pklFile)

        #Leader
        leaderNpTimePosObss = np.array(leaderFollowerTimeRangesDict["leaderTimeRangesObss"])[0:rowsNum]
        leaderNpNormalPosObss = RowsNormalizer.getNpNormalizedNpRows(leaderNpTimePosObss[0:, 1:])
        leaderNpTimeRows = leaderNpTimePosObss[0:,0:1]
        leaderEncoderModel = load_model(sharedPathToTwoLidars+"autoencoders/leader-encoder-rows-num-50000-epochs-2160-batch-size-32.h5")
        leaderLowDimPos = leaderEncoderModel(leaderNpNormalPosObss)
        leaderLowDimTimePosObss = np.hstack((leaderNpTimeRows, leaderLowDimPos))
        leaderLowDimTimePosVelObss = TimePosRowsDerivativeComputer.computer(leaderLowDimTimePosObss, velCoefficient)
        leaderLowDimPosVelObss = leaderLowDimTimePosVelObss[0:,1:]
        Plots.plot3DEncodedXTrain(leaderLowDimPos)

        # Follower
        followerNpTimePosObss = np.array(leaderFollowerTimeRangesDict["followerTimeRangesObss"])[0:rowsNum]
        followerNpNormalPosObss = RowsNormalizer.getNpNormalizedNpRows(followerNpTimePosObss[0:, 1:])
        followerNpTimeRows = followerNpTimePosObss[0:,0:1]
        followerEncoderModel = load_model(sharedPathToTwoLidars+"autoencoders/follower-encoder-rows-num-50000-epochs-2160-batch-size-32.h5")
        followerLowDimPos = followerEncoderModel(followerNpNormalPosObss)
        followerLowDimTimePosObss = np.hstack((followerNpTimeRows, followerLowDimPos))
        followerLowDimTimePosVelObss = TimePosRowsDerivativeComputer.computer(followerLowDimTimePosObss, velCoefficient)
        followerLowDimPosVelObss = followerLowDimTimePosVelObss[0:,1:]
        Plots.plot3DEncodedXTrain(followerLowDimPos)

        '''Cluster each'''

        leaderTimePosVelClusteringStrgy = PosVelsClusteringStrgy(leaderClustersNum
                                                                 , leaderLowDimPosVelObss)

        followerTimePosVelClusteringStrgy = PosVelsClusteringStrgy(followerClustersNum
                                                                   , followerLowDimPosVelObss)
        '''Build transition matrix'''
        transitionMatrix = TwoAlphabetWordsTransitionMatrix(leaderTimePosVelClusteringStrgy
                                                            , followerTimePosVelClusteringStrgy
                                                            , leaderLowDimTimePosVelObss
                                                            , followerLowDimTimePosVelObss)
        # transitionMatrix.setLeaderFollowerObsMatchStrgy("ALREADY_INDEX_MATCHED")
        transitionMatrix.save(sharedPathToTwoLidars + "autoencoders/transtionMatrix-{}*{}.txt".format(leaderClustersNum, followerClustersNum))



if __name__ == "__main__":
    NormalScenarioTransitionMatrixBuilderBooter.boot()
    os.system('spd-say "Transition matrix build and saved"')