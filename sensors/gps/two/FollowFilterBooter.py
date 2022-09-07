from ctumrs.ClusterLevelAbnormalVals import ClusterLevelAbnormalVals
from ctumrs.TimePosVelObssUtility import TimePosVelObssUtility
from ctumrs.TwoAlphabetWordsTransitionMatrix import TwoAlphabetWordsTransitionMatrix
from ctumrs.TimePosVelObssPlottingUtility import TimePosVelObssPlottingUtility

plottingUtility = TimePosVelObssPlottingUtility()
timePosVelObssUtility = TimePosVelObssUtility()

leaderUavClustersNum = 75
followerUavClustersNum = 75
velocityCoefficient = 10000

jointPathToLeaderAndFollowerNormalScenario= "/home/donkarlo/Dropbox/projs/research/data/self-aware-drones/ctumrs/two-drones/normal-scenario/gps/"
'''Lodaing the transition matrix'''
jointfilePathToTransitionMatrix = jointPathToLeaderAndFollowerNormalScenario + "transtionMatrix-{}*{}.txt".format(leaderUavClustersNum, followerUavClustersNum)
transitionMatrix = TwoAlphabetWordsTransitionMatrix()
transitionMatrix = transitionMatrix.load(jointfilePathToTransitionMatrix)

''''''
jointPathToLeaderAndFollowerAbnormalScenario= "/home/donkarlo/Dropbox/projs/research/data/self-aware-drones/ctumrs/two-drones/follow-scenario/gps/"
pathToLeaderUavTimePosVelDataFile = jointPathToLeaderAndFollowerAbnormalScenario + "uav-1-cleaned-gps-pos-vel.txt"
pathToFollowerUavTimePosVelDataFile = jointPathToLeaderAndFollowerAbnormalScenario + "uav-2-cleaned-gps-pos-vel.txt"

leaderUavPosVelsAndTimePosVels = timePosVelObssUtility.getTimePosVelsAndPosVels(pathToLeaderUavTimePosVelDataFile, velocityCoefficient)
leaderPosVelObss = leaderUavPosVelsAndTimePosVels['posVels']

followerUavPosVelsAndTimePosVels = timePosVelObssUtility.getTimePosVelsAndPosVels(pathToFollowerUavTimePosVelDataFile, velocityCoefficient)
followerPosVelObss = followerUavPosVelsAndTimePosVels['posVels']


leaderFollowerFilter = ClusterLevelAbnormalVals(transitionMatrix)
noveltyValues = leaderFollowerFilter.getClusterLevelAbnormalValsByPosVelsObss(leaderPosVelObss
                                                                              , followerPosVelObss)
leaderFollowerFilter.plotNovelties(noveltyValues)