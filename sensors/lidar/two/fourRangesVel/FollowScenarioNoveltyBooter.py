import pickle

from ctumrs.ClusterLevelAbnormalVals import ClusterLevelAbnormalVals
from ctumrs.TransitionMatrix import TransitionMatrix
from ctumrs.sensors.lidar.two.fourRangesVel.TimeFourRangesVelsObss import TimeFourRangesVelsObss
from ctumrs.sensors.lidar.two.rangeSumVel.TimeRangeSumVelObss import TimeRangeSumVelObss

leaderClustersNum = 75
followerClustersNum = 75
velCoefficient = 10000

sharedPathToLeaderAndFollowerNormalScenario= "/home/donkarlo/Dropbox/projs/research/data/self-aware-drones/ctumrs/two-drones/normal-scenario/lidar/fourRangesVels/"
'''Lodaing the transition matrix'''
jointfilePathToTransitionMatrix = sharedPathToLeaderAndFollowerNormalScenario + "transtionMatrix-{}*{}.txt".format(leaderClustersNum, followerClustersNum)
transitionMatrix = TransitionMatrix()
transitionMatrix = transitionMatrix.load(jointfilePathToTransitionMatrix)

''''''
pklFile = open("/home/donkarlo/Dropbox/projs/research/data/self-aware-drones/ctumrs/two-drones/follow-scenario/lidar/fourRangesVels/"+"twoLidarsTimeFourRangesVelsObss.pkl", "rb")
leaderFollowerTimeRangeSumVelDict = pickle.load(pklFile)

leaderPosVelObss = TimeFourRangesVelsObss.velMulInRangeSumVelObss(leaderFollowerTimeRangeSumVelDict['leaderFourRangesVelsObss'], velCoefficient)
followerPosVelObss = TimeFourRangesVelsObss.velMulInRangeSumVelObss(leaderFollowerTimeRangeSumVelDict['followerFourRangesVelsObss'], velCoefficient)


leaderFollowerFilter = ClusterLevelAbnormalVals(transitionMatrix)
noveltyValues = leaderFollowerFilter.getClusterLevelAbnormalValsByPosVelsObss(leaderPosVelObss
                                                                              , followerPosVelObss)
leaderFollowerFilter.plotNovelties(noveltyValues)