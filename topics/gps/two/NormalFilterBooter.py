from ctumrs.LeaderFollowerFilter import LeaderFollowerFilter
from ctumrs.TimePosVelObssUtility import TimePosVelObssUtility
from ctumrs.TransitionMatrix import TransitionMatrix
from ctumrs.TimePosVelObssPlottingUtility import TimePosVelObssPlottingUtility

plottingUtility = TimePosVelObssPlottingUtility()
timePosVelObssUtility = TimePosVelObssUtility()

leaderUavClustersNum = 75
followerUavClustersNum = 75
velocityCoefficient = 10000

jointPathToLeaderAndFollowerNormalScenario= "/home/donkarlo/Dropbox/projs/research/data/self-aware-drones/ctumrs/two-drones/normal-scenario/gps/"
'''Lodaing the transition matrix'''
jointfilePathToTransitionMatrix = jointPathToLeaderAndFollowerNormalScenario + "transtionMatrix-{}*{}.txt".format(leaderUavClustersNum, followerUavClustersNum)
transitionMatrix = TransitionMatrix()
transitionMatrix = transitionMatrix.load(jointfilePathToTransitionMatrix)

''''''
jointPathToLeaderAndFollowerAbnormalScenario= "/home/donkarlo/Dropbox/projs/research/data/self-aware-drones/ctumrs/two-drones/normal-scenario/gps/"
pathToLeaderUavTimePosVelDataFile = jointPathToLeaderAndFollowerAbnormalScenario + "uav-1-cleaned-gps-pos-vel.txt"
pathToFollowerUavTimePosVelDataFile = jointPathToLeaderAndFollowerAbnormalScenario + "uav-2-cleaned-gps-pos-vel.txt"

leaderUavPosVelsAndTimePosVels = timePosVelObssUtility.getTimePosVelsAndPosVels(pathToLeaderUavTimePosVelDataFile, velocityCoefficient)
leaderPosVelObss = leaderUavPosVelsAndTimePosVels['posVels']

followerUavPosVelsAndTimePosVels = timePosVelObssUtility.getTimePosVelsAndPosVels(pathToFollowerUavTimePosVelDataFile, velocityCoefficient)
followerPosVelObss = followerUavPosVelsAndTimePosVels['posVels']


leaderFollowerFilter = LeaderFollowerFilter(transitionMatrix)
noveltyValues = leaderFollowerFilter.getPosVelsObssNovelties(leaderPosVelObss
                                             ,followerPosVelObss)
leaderFollowerFilter.plotNovelties(noveltyValues)