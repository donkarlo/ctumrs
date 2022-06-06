import pickle

import numpy as np

from ctumrs.LeaderFollowerFilter import LeaderFollowerFilter
from ctumrs.TransitionMatrix import TransitionMatrix
from ctumrs.topics.rplidar.twoRpLidar.ranges.TimeRangesVelsObss import TimeRangesVelsObss

leaderClustersNum = 75
followerClustersNum = 75
velCoefficient = 10000

sharedPathToLeaderAndFollowerNormalScenario= "/home/donkarlo/Dropbox/projs/research/data/self-aware-drones/ctumrs/two-drones/normal-scenario/lidars/ranges/"
'''Lodaing the transition matrix'''
jointfilePathToTransitionMatrix = sharedPathToLeaderAndFollowerNormalScenario + "transtionMatrix-{}*{}.txt".format(leaderClustersNum, followerClustersNum)
transitionMatrix = TransitionMatrix()
transitionMatrix = transitionMatrix.load(jointfilePathToTransitionMatrix)

''''''
pklFile = open("/home/donkarlo/Dropbox/projs/research/data/self-aware-drones/ctumrs/two-drones/follow-scenario/lidars/ranges/"+"twoLidarsTimeRangesVelsObss.pkl", "rb")
leaderFollowerTimeRangeSumVelDict = pickle.load(pklFile)

leaderPosVelObss = TimeRangesVelsObss.velMulInTimeRangesVelsObss(np.array(leaderFollowerTimeRangeSumVelDict['leaderTimeRangesVelsObss']), velCoefficient)
followerPosVelObss = TimeRangesVelsObss.velMulInTimeRangesVelsObss(np.array(leaderFollowerTimeRangeSumVelDict['followerTimeRangesVelsObss']), velCoefficient)


leaderFollowerFilter = LeaderFollowerFilter(transitionMatrix)
noveltyValues = leaderFollowerFilter.getPosVelsObssNovelties(leaderPosVelObss[:,1:]
                                             ,followerPosVelObss[:,1:])
leaderFollowerFilter.plotNovelties(noveltyValues)