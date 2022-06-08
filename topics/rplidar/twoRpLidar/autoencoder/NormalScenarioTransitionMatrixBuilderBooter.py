import os
import pickle

import numpy as np

from MachineSettings import MachineSettings
from ctumrs.TimePosVelsClusteringStrgy import TimePosVelsClusteringStrgy
from ctumrs.TransitionMatrix import TransitionMatrix
from keras.models import load_model
from ctumrs.topics.rplidar.twoRpLidar.autoencoder.Plots import Plots
from mMath.data.RowsTimeDerivativeComputer import RowsTimeDerivativeComputer
from mMath.data.preProcess.RowsNormalizer import RowsNormalizer

class NormalScenarioTransitionMatrixBuilderBooter:
    @staticmethod
    def boot():
        sharedPathToTwoLidars = MachineSettings.MAIN_PATH+"projs/research/data/self-aware-drones/ctumrs/two-drones/normal-scenario/lidars/"
        pickleFileName = "twoLidarsTimeRangesObss.pkl"

        rowsNum = 50000
        velCoefficient = 10000
        leaderClustersNum = 75
        followerClustersNum = 75

        '''Load data'''
        pklFile = open(sharedPathToTwoLidars + pickleFileName, "rb")
        leaderFollowerTimeRangesDict = pickle.load(pklFile)

        #Leader
        leaderNpTimePosObss = np.array(leaderFollowerTimeRangesDict["leaderTimeRangesObss"])[0:rowsNum]
        leaderNpNormalPosObss = RowsNormalizer.getNpNormalizedNpRows(leaderNpTimePosObss[0:, 1:])
        leaderNpTimeRows = leaderNpTimePosObss[0:,0:1]
        leaderEncoderModel = load_model(sharedPathToTwoLidars+"autoencoders/leader-encoder.h5")
        leaderLowDimPos = leaderEncoderModel(leaderNpNormalPosObss)
        leaderLowDimTimePosObss = np.hstack((leaderNpTimeRows, leaderLowDimPos))
        leaderLowDimTimePosVelObss = RowsTimeDerivativeComputer.computer(leaderLowDimTimePosObss,velCoefficient)
        leaderLowDimPosVelObss = leaderLowDimTimePosVelObss[0:,1:]
        Plots.plot2DEncodedXTrain(leaderLowDimPos)

        # Follower
        followerNpTimePosObss = np.array(leaderFollowerTimeRangesDict["followerTimeRangesObss"])[0:rowsNum]
        followerNpNormalPosObss = RowsNormalizer.getNpNormalizedNpRows(followerNpTimePosObss[0:, 1:])
        followerNpTimeRows = followerNpTimePosObss[0:,0:1]
        followerEncoderModel = load_model(sharedPathToTwoLidars+"autoencoders/follower-encoder.h5")
        followerLowDimPos = followerEncoderModel(followerNpNormalPosObss)
        followerLowDimTimePosObss = np.hstack((followerNpTimeRows, followerLowDimPos))
        followerLowDimTimePosVelObss = RowsTimeDerivativeComputer.computer(followerLowDimTimePosObss,velCoefficient)
        followerLowDimPosVelObss = followerLowDimTimePosVelObss[0:,1:]
        Plots.plot2DEncodedXTrain(followerLowDimPos)

        '''Cluster each'''

        leaderTimePosVelClusteringStrgy = TimePosVelsClusteringStrgy(leaderClustersNum
                                                                     , leaderLowDimTimePosVelObss
                                                                     , leaderLowDimPosVelObss)

        followerTimePosVelClusteringStrgy = TimePosVelsClusteringStrgy(followerClustersNum
                                                                       , followerLowDimTimePosVelObss
                                                                       , followerLowDimPosVelObss)
        '''Build transition matrix'''
        transitionMatrix = TransitionMatrix(leaderTimePosVelClusteringStrgy
                                            , followerTimePosVelClusteringStrgy
                                            , leaderLowDimTimePosVelObss
                                            , followerLowDimTimePosVelObss)
        # transitionMatrix.setLeaderFollowerObsMatchStrgy("ALREADY_INDEX_MATCHED")
        transitionMatrix.save(sharedPathToTwoLidars + "autoencoders/transtionMatrix-{}*{}.txt".format(leaderClustersNum, followerClustersNum))



if __name__ == "__main__":
    NormalScenarioTransitionMatrixBuilderBooter.boot()
    os.system('spd-say "Transition matrix build and saved"')