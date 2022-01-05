from ctumrs.LeaderFollowerFilter import LeaderFollowerFilter
from ctumrs.TransitionMatrix import TransitionMatrix
from ctumrs.Utilities import *

leaderUavClustersNum = 8
followerUavClustersNum = 8
velocityCoefficient = 1
jointPathToLeaderAndFollower= "/home/donkarlo/Dropbox/projs/research/data/self-aware-drones/ctumrs/two-drones/inner-squares/"

'''Lodaing the transition matric'''
jointfilePathToTransitionMatrix = jointPathToLeaderAndFollower+"transtionMatrix-{}*{}.txt".format(leaderUavClustersNum,followerUavClustersNum)
transitionMatrix = TransitionMatrix()
transitionMatrix.loadNpTransitionMatrix(jointfilePathToTransitionMatrix)

''''''
pathToLeaderUavTimePosVelDataFile = jointPathToLeaderAndFollower+"gps-uav1-pos-vel.txt"
pathToFollowerUavTimePosVelDataFile = jointPathToLeaderAndFollower+"gps-uav2-pos-vel.txt"
leaderUavPosVelsAndTimePosVels = getTimePosVelsAndPosVels(pathToLeaderUavTimePosVelDataFile, velocityCoefficient)
leaderPosVelObss = leaderUavPosVelsAndTimePosVels['posVels']
followerUavPosVelsAndTimePosVels = getTimePosVelsAndPosVels(pathToFollowerUavTimePosVelDataFile, velocityCoefficient)
followerPosVelObss = followerUavPosVelsAndTimePosVels['posVels']


leaderFollowerFilter = LeaderFollowerFilter(transitionMatrix)
leaderFollowerFilter.getPosVelsObssNovelties(leaderPosVelObss
                                             ,followerPosVelObss)