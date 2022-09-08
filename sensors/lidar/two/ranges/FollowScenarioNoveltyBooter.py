import pickle

import numpy as np

from ctumrs.TwoAlphabetWordsClusterLevelAbnormalVals import TwoAlphabetWordsClusterLevelAbnormalVals
from ctumrs.TwoAlphabetWordsTransitionMatrix import TwoAlphabetWordsTransitionMatrix
from ctumrs.sensors.lidar.two.ranges.TimeRangesVelsObss import TimeRangesVelsObss
from MachineSettings import MachineSettings

leaderClustersNum = 75
followerClustersNum = 75
velCoefficient = 20

sharedPathToLeaderAndFollowerNormalScenario= "{}projs/research/data/self-aware-drones/ctumrs/two-drones/normal-scenario/lidar/allRanges/".format(MachineSettings.MAIN_PATH)
'''Lodaing the transition matrix'''
jointfilePathToTransitionMatrix = sharedPathToLeaderAndFollowerNormalScenario + "transtionMatrix-clusters-{}-{}-velco-{}.txt".format(leaderClustersNum, followerClustersNum, velCoefficient)
transitionMatrix = TwoAlphabetWordsTransitionMatrix()
transitionMatrix = transitionMatrix.load(jointfilePathToTransitionMatrix)

''''''
pklFile = open("{}projs/research/data/self-aware-drones/ctumrs/two-drones/follow-scenario/lidar/allRanges/".format(MachineSettings.MAIN_PATH)+"twoLidarsTimeRangesVelsObss.pkl", "rb")
leaderFollowerTimeRangeSumVelDict = pickle.load(pklFile)

leaderPosVelObss = TimeRangesVelsObss.velMulInTimeRangesVelsObss(np.array(leaderFollowerTimeRangeSumVelDict['leaderTimeRangesVelsObss']), velCoefficient)
followerPosVelObss = TimeRangesVelsObss.velMulInTimeRangesVelsObss(np.array(leaderFollowerTimeRangeSumVelDict['followerTimeRangesVelsObss']), velCoefficient)


leaderFollowerFilter = TwoAlphabetWordsClusterLevelAbnormalVals(transitionMatrix)
noveltyValues = leaderFollowerFilter.getClusterLevelAbnormalValsByPosVelsObss(leaderPosVelObss[:, 1:]
                                                                              ,followerPosVelObss[:,1:])
leaderFollowerFilter.plotNovelties(noveltyValues)