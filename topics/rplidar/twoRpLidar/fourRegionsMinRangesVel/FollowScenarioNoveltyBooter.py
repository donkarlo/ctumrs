import pickle

from ctumrs.LeaderFollowerFilter import LeaderFollowerFilter
from ctumrs.TransitionMatrix import TransitionMatrix
from ctumrs.topics.rplidar.twoRpLidar.fourRegionsMinRangesVel.TimeFourRegionsMinVelsObss import \
    TimeFourRegionsMinRangesVelsObss

leaderClustersNum = 75
followerClustersNum = 75
velCoefficient = 10000

sharedPathToLeaderAndFollowerNormalScenario= "/home/donkarlo/Dropbox/projs/research/data/self-aware-drones/ctumrs/two-drones/normal-scenario/lidars/fourRegionsMinRangesVels/"
'''Lodaing the transition matrix'''
jointfilePathToTransitionMatrix = sharedPathToLeaderAndFollowerNormalScenario + "transtionMatrix-{}*{}.txt".format(leaderClustersNum, followerClustersNum)
transitionMatrix = TransitionMatrix()
transitionMatrix = transitionMatrix.load(jointfilePathToTransitionMatrix)

''''''
pklFile = open("/home/donkarlo/Dropbox/projs/research/data/self-aware-drones/ctumrs/two-drones/follow-scenario/lidars/fourRegionsMinRangesVels/"+"twoLidarsTimeFourRegionsMinRangesVelsObss.pkl", "rb")
leaderFollowerTimeRangeSumVelDict = pickle.load(pklFile)

leaderPosVelObss = TimeFourRegionsMinRangesVelsObss.velMulInRangeSumVelObss(leaderFollowerTimeRangeSumVelDict['leaderFourRegionsMinRangesVelsObss'], velCoefficient)
followerPosVelObss = TimeFourRegionsMinRangesVelsObss.velMulInRangeSumVelObss(leaderFollowerTimeRangeSumVelDict['followerFourRegionsMinRangesVelsObss'], velCoefficient)


leaderFollowerFilter = LeaderFollowerFilter(transitionMatrix)
noveltyValues = leaderFollowerFilter.getPosVelsObssNovelties(leaderPosVelObss
                                             ,followerPosVelObss)
leaderFollowerFilter.plotNovelties(noveltyValues)