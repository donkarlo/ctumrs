import pickle

from ctumrs.ClusterLevelAbnormalVals import ClusterLevelAbnormalVals
from ctumrs.TwoAlphabetWordsTransitionMatrix import TwoAlphabetWordsTransitionMatrix
from ctumrs.sensors.lidar.two.rangeSumVel.TimeRangeSumVelObss import TimeRangeSumVelObss

leaderUavClustersNum = 75
followerUavClustersNum = 75
velCoefficient = 10000

jointPathToLeaderAndFollowerNormalScenario= "/home/donkarlo/Dropbox/projs/research/data/self-aware-drones/ctumrs/two-drones/normal-scenario/lidar/rangeSumVel/"
'''Lodaing the transition matrix'''
jointfilePathToTransitionMatrix = jointPathToLeaderAndFollowerNormalScenario + "transtionMatrix-{}*{}.txt".format(leaderUavClustersNum, followerUavClustersNum)
transitionMatrix = TwoAlphabetWordsTransitionMatrix()
transitionMatrix = transitionMatrix.load(jointfilePathToTransitionMatrix)

''''''
pklFile = open(jointPathToLeaderAndFollowerNormalScenario+"twoLidarsTimeRangeSumVelObss.pkl", "rb")
leaderFollowerTimeRangeSumVelDict = pickle.load(pklFile)


leaderPosVelObss = TimeRangeSumVelObss.velMulInRangeSumVelObss(leaderFollowerTimeRangeSumVelDict['leaderRangeSumVelObss'], velCoefficient)
followerPosVelObss = TimeRangeSumVelObss.velMulInRangeSumVelObss(leaderFollowerTimeRangeSumVelDict['followerRangeSumVelObss'], velCoefficient)


leaderFollowerFilter = ClusterLevelAbnormalVals(transitionMatrix)
noveltyValues = leaderFollowerFilter.getClusterLevelAbnormalValsByPosVelsObss(leaderPosVelObss
                                                                              , followerPosVelObss)
leaderFollowerFilter.plotNovelties(noveltyValues)