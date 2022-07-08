from ctumrs.ClusterLevelAbnormalVals import ClusterLevelAbnormalVals
from ctumrs.TransitionMatrix import TransitionMatrix
from ctumrs.TimePosVelObssPlottingUtility import TimePosVelObssPlottingUtility

utility = TimePosVelObssPlottingUtility()

leaderUavClustersNum = 75
followerUavClustersNum = 75
velocityCoefficient = 10000

jointPathToLeaderAndFollowerNormalScenario= "/home/donkarlo/Dropbox/projs/research/data/self-aware-drones/ctumrs/two-drones/inner-squares/"
'''Lodaing the transition matrix'''
jointfilePathToTransitionMatrix = jointPathToLeaderAndFollowerNormalScenario + "transtionMatrix-{}*{}.txt".format(leaderUavClustersNum, followerUavClustersNum)
transitionMatrix = TransitionMatrix()
transitionMatrix = transitionMatrix.load(jointfilePathToTransitionMatrix)

''''''
jointPathToLeaderAndFollowerAbnormalScenario= "/home/donkarlo/Dropbox/projs/research/data/self-aware-drones/ctumrs/two-drones/inner-squares-decrease-distance-2-p-7/"
pathToLeaderUavTimePosVelDataFile = jointPathToLeaderAndFollowerAbnormalScenario + "uav-1-cleaned-gps-pos-vel.txt"
pathToFollowerUavTimePosVelDataFile = jointPathToLeaderAndFollowerAbnormalScenario + "uav-2-cleaned-gps-pos-vel.txt"

leaderUavPosVelsAndTimePosVels = utility.getTimePosVelsAndPosVels(pathToLeaderUavTimePosVelDataFile, velocityCoefficient)
leaderPosVelObss = leaderUavPosVelsAndTimePosVels['posVels']

followerUavPosVelsAndTimePosVels = utility.getTimePosVelsAndPosVels(pathToFollowerUavTimePosVelDataFile, velocityCoefficient)
followerPosVelObss = followerUavPosVelsAndTimePosVels['posVels']


leaderFollowerFilter = ClusterLevelAbnormalVals(transitionMatrix)
noveltyValues = leaderFollowerFilter.getClusterLevelAbnormalValsByPosVelsObss(leaderPosVelObss
                                                                              , followerPosVelObss)
leaderFollowerFilter.plotNovelties(noveltyValues)