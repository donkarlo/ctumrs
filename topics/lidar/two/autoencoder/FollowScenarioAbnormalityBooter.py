import pickle

import numpy as np

from ctumrs.LeaderFollowerFilter import LeaderFollowerFilter
from MachineSettings import MachineSettings
from ctumrs.TransitionMatrix import TransitionMatrix
from mMath.data.RowsTimeDerivativeComputer import RowsTimeDerivativeComputer
from mMath.data.preProcess.RowsNormalizer import RowsNormalizer
from keras.models import load_model


class FollowScenarioAbnormalityBooter:
    @staticmethod
    def boot():
        #Settings
        rowsNum = 50000
        leaderClustersNum = 75
        followerClustersNum = 75
        velCoefficient = 10000
        sharedPathToTwoDronesData = MachineSettings.MAIN_PATH+"projs/research/data/self-aware-drones/ctumrs/two-drones/"
        sharedPathToLeaderAndFollowerNormalScenario= sharedPathToTwoDronesData+"normal-scenario/lidar/"

        '''Lodaing the transition matrix'''
        jointfilePathToTransitionMatrix = sharedPathToLeaderAndFollowerNormalScenario + "autoencoders/transtionMatrix-{}*{}.txt".format(leaderClustersNum, followerClustersNum)
        transitionMatrix = TransitionMatrix()
        transitionMatrix = transitionMatrix.load(jointfilePathToTransitionMatrix)

        '''Loading follow scenrio data'''
        pklFile = open(sharedPathToTwoDronesData+"follow-scenario/lidar/"+"twoLidarsTimeRangesObss.pkl", "rb")
        leaderFollowerTimeRangesDict = pickle.load(pklFile)

        # Leader
        leaderNpTimePosObss = np.array(leaderFollowerTimeRangesDict["leaderTimeRangesObss"])[0:rowsNum]
        leaderNpNormalPosObss = RowsNormalizer.getNpNormalizedNpRows(leaderNpTimePosObss[0:, 1:])
        leaderNpTimeRows = leaderNpTimePosObss[0:, 0:1]
        leaderEncoderModel = load_model(sharedPathToLeaderAndFollowerNormalScenario + "autoencoders/leader-encoder-rows-num-50000-epochs-2160-batch-size-32.h5")
        leaderLowDimPos = leaderEncoderModel(leaderNpNormalPosObss)
        leaderLowDimTimePosObss = np.hstack((leaderNpTimeRows, leaderLowDimPos))
        leaderLowDimTimePosVelObss = RowsTimeDerivativeComputer.computer(leaderLowDimTimePosObss, velCoefficient)
        leaderLowDimPosVelObss = leaderLowDimTimePosVelObss[0:, 1:]

        # Follower
        followerNpTimePosObss = np.array(leaderFollowerTimeRangesDict["followerTimeRangesObss"])[0:rowsNum]
        followerNpNormalPosObss = RowsNormalizer.getNpNormalizedNpRows(followerNpTimePosObss[0:, 1:])
        followerNpTimeRows = followerNpTimePosObss[0:, 0:1]
        followerEncoderModel = load_model(sharedPathToLeaderAndFollowerNormalScenario + "autoencoders/follower-encoder-rows-num-50000-epochs-2160-batch-size-32.h5")
        followerLowDimPos = followerEncoderModel(followerNpNormalPosObss)
        followerLowDimTimePosObss = np.hstack((followerNpTimeRows, followerLowDimPos))
        followerLowDimTimePosVelObss = RowsTimeDerivativeComputer.computer(followerLowDimTimePosObss, velCoefficient)
        followerLowDimPosVelObss = followerLowDimTimePosVelObss[0:, 1:]


        leaderFollowerFilter = LeaderFollowerFilter(transitionMatrix)
        noveltyValues = leaderFollowerFilter.getPosVelsObssNovelties(leaderLowDimPosVelObss, followerLowDimPosVelObss)
        leaderFollowerFilter.plotNovelties(noveltyValues)

if __name__ == "__main__":
    FollowScenarioAbnormalityBooter.boot()