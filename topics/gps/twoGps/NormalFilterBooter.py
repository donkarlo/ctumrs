from ctumrs.LeaderFollowerFilter import LeaderFollowerFilter
from ctumrs.TwoAlphabetWordsTransitionMatrix import TwoAlphabetWordsTransitionMatrix
from ctumrs.TimePosVelObssPlottingUtility import TimePosVelObssPlottingUtility

utility = TimePosVelObssPlottingUtility()

leaderUavClustersNum = 75
followerUavClustersNum = 75
velocityCoefficient = 10000

jointPathToLeaderAndFollowerNormalScenario= "/home/donkarlo/Dropbox/projs/research/data/self-aware-drones/ctumrs/two-drones/normal-scenario/gps/"
'''Lodaing the transition matrix'''
jointfilePathToTransitionMatrix = jointPathToLeaderAndFollowerNormalScenario + "transtionMatrix-{}*{}.txt".format(leaderUavClustersNum, followerUavClustersNum)
transitionMatrix = TwoAlphabetWordsTransitionMatrix()
transitionMatrix = transitionMatrix.load(jointfilePathToTransitionMatrix)

''''''
jointPathToLeaderAndFollowerAbnormalScenario= "/home/donkarlo/Dropbox/projs/research/data/self-aware-drones/ctumrs/two-drones/normal-scenario/gps/"
pathToLeaderUavTimePosVelDataFile = jointPathToLeaderAndFollowerAbnormalScenario + "uav-1-cleaned-gps-pos-vel.txt"
pathToFollowerUavTimePosVelDataFile = jointPathToLeaderAndFollowerAbnormalScenario + "uav-2-cleaned-gps-pos-vel.txt"

leaderUavPosVelsAndTimePosVels = utility.getTimePosVelsAndPosVels(pathToLeaderUavTimePosVelDataFile, velocityCoefficient)
leaderPosVelObss = leaderUavPosVelsAndTimePosVels['posVels']

followerUavPosVelsAndTimePosVels = utility.getTimePosVelsAndPosVels(pathToFollowerUavTimePosVelDataFile, velocityCoefficient)
followerPosVelObss = followerUavPosVelsAndTimePosVels['posVels']


leaderFollowerFilter = LeaderFollowerFilter(transitionMatrix)
noveltyValues = leaderFollowerFilter.getPosVelsObssNovelties(leaderPosVelObss
                                             ,followerPosVelObss)
leaderFollowerFilter.plotNovelties(noveltyValues)