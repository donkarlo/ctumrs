import pickle

from ctumrs.LeaderFollowerFilter import LeaderFollowerFilter
from ctumrs.TransitionMatrix import TransitionMatrix
from ctumrs.topics.lidar.two.rangeSumVel.TimeRangeSumVelObss import TimeRangeSumVelObss

leaderUavClustersNum = 75
followerUavClustersNum = 75
velCoefficient = 10000

jointPathToLeaderAndFollowerNormalScenario= "/home/donkarlo/Dropbox/projs/research/data/self-aware-drones/ctumrs/two-drones/normal-scenario/lidars/rangeSumVel/"
'''Lodaing the transition matrix'''
jointfilePathToTransitionMatrix = jointPathToLeaderAndFollowerNormalScenario + "transtionMatrix-{}*{}.txt".format(leaderUavClustersNum, followerUavClustersNum)
transitionMatrix = TransitionMatrix()
transitionMatrix = transitionMatrix.load(jointfilePathToTransitionMatrix)

''''''
pklFile = open(jointPathToLeaderAndFollowerNormalScenario+"twoLidarsTimeRangeSumVelObss.pkl", "rb")
leaderFollowerTimeRangeSumVelDict = pickle.load(pklFile)


leaderPosVelObss = TimeRangeSumVelObss.velMulInRangeSumVelObss(leaderFollowerTimeRangeSumVelDict['leaderRangeSumVelObss'], velCoefficient)
followerPosVelObss = TimeRangeSumVelObss.velMulInRangeSumVelObss(leaderFollowerTimeRangeSumVelDict['followerRangeSumVelObss'], velCoefficient)


leaderFollowerFilter = LeaderFollowerFilter(transitionMatrix)
noveltyValues = leaderFollowerFilter.getPosVelsObssNovelties(leaderPosVelObss
                                             ,followerPosVelObss)
leaderFollowerFilter.plotNovelties(noveltyValues)