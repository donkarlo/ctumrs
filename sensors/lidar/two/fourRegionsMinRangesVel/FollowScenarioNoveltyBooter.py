import pickle

from ctumrs.ClusterLevelAbnormalVals import ClusterLevelAbnormalVals
from ctumrs.TransitionMatrix import TransitionMatrix
from ctumrs.sensors.lidar.two.fourRegionsMinRangesVel.TimeFourRegionsMinVelsObss import \
    TimeFourRegionsMinRangesVelsObss

leaderClustersNum = 75
followerClustersNum = 75
velCoefficient = 10000

sharedPathToLeaderAndFollowerNormalScenario= "/home/donkarlo/Dropbox/projs/research/data/self-aware-drones/ctumrs/two-drones/normal-scenario/lidar/fourRegionsMinRangesVels/"
'''Lodaing the transition matrix'''
jointfilePathToTransitionMatrix = sharedPathToLeaderAndFollowerNormalScenario + "transtionMatrix-{}*{}.txt".format(leaderClustersNum, followerClustersNum)
transitionMatrix = TransitionMatrix()
transitionMatrix = transitionMatrix.load(jointfilePathToTransitionMatrix)

''''''
pklFile = open("/home/donkarlo/Dropbox/projs/research/data/self-aware-drones/ctumrs/two-drones/follow-scenario/lidar/fourRegionsMinRangesVels/"+"twoLidarsTimeFourRegionsMinRangesVelsObss.pkl", "rb")
leaderFollowerTimeRangeSumVelDict = pickle.load(pklFile)

leaderPosVelObss = TimeFourRegionsMinRangesVelsObss.velMulInRangeSumVelObss(leaderFollowerTimeRangeSumVelDict['leaderFourRegionsMinRangesVelsObss'], velCoefficient)
followerPosVelObss = TimeFourRegionsMinRangesVelsObss.velMulInRangeSumVelObss(leaderFollowerTimeRangeSumVelDict['followerFourRegionsMinRangesVelsObss'], velCoefficient)


leaderFollowerFilter = ClusterLevelAbnormalVals(transitionMatrix)
noveltyValues = leaderFollowerFilter.getClusterLevelAbnormalValsByPosVelsObss(leaderPosVelObss
                                                                              , followerPosVelObss)
leaderFollowerFilter.plotNovelties(noveltyValues)