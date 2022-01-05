from ctumrs.TransitionMatrix import TransitionMatrix
from ctumrs.Utilities import *
from ctumrs.TimePosVelsClusteringStrgy import TimePosVelsClusteringStrgy
from ctumrs.LeaderFollowerFilter import LeaderFollowerFilter

leaderClustersNum = 8
followerClustersNum = 8
velocityCoefficient = 10000

jointPathToLeaderAndFollower= "/home/donkarlo/Dropbox/projs/research/data/self-aware-drones/ctumrs/two-drones/inner-squares/"
pathToLeaderUavTimePosVelDataFile = jointPathToLeaderAndFollower+"gps-uav1-pos-vel.txt"
pathToFollowerUavTimePosVelDataFile = jointPathToLeaderAndFollower+"gps-uav2-pos-vel.txt"

leaderUavPosVelsAndTimePosVels = getTimePosVelsAndPosVels(pathToLeaderUavTimePosVelDataFile, velocityCoefficient)
leaderPosVels = leaderUavPosVelsAndTimePosVels['posVels']
leaderTimePosVels = leaderUavPosVelsAndTimePosVels['timePosVels']

followerUavPosVelsAndTimePosVels = getTimePosVelsAndPosVels(pathToFollowerUavTimePosVelDataFile, velocityCoefficient)
followerPosVels = followerUavPosVelsAndTimePosVels['posVels']
followerTimePosVels = followerUavPosVelsAndTimePosVels['timePosVels']

############PLOTTING THE TRAJECTORY##############
plotPos(leaderPosVels)
plotPos(followerPosVels)

############CLUSTERING###########
leaderTimePosVelClusters = TimePosVelsClusteringStrgy(leaderClustersNum
                                                      , leaderTimePosVels
                                                      , leaderPosVels)
leaderTimePosVelClustersDict = leaderTimePosVelClusters.getLabeledTimePosVelsClustersDict()

followerTimePosVelClusters = TimePosVelsClusteringStrgy(followerClustersNum
                                                        , followerTimePosVels
                                                        , followerPosVels)
followerTimePosVelClustersDict = followerTimePosVelClusters.getLabeledTimePosVelsClustersDict()

'''PLOTING THE CLUSTERS'''
plotPosWithCLusters(leaderTimePosVelClustersDict)
plotPosWithCLusters(followerTimePosVelClustersDict)
plotLeaderFollowerUavPosWithCLusters(leaderTimePosVelClustersDict
                                     , followerTimePosVelClustersDict)

'''
Building the transition matrix
'''
transitionMatrix = TransitionMatrix(leaderTimePosVelClusters
                                    , followerTimePosVelClusters
                                    , leaderTimePosVels
                                    , followerTimePosVels)
transitionMatrix.saveNpTransitionMatrix(jointPathToLeaderAndFollower +"transtionMatrix-{}*{}.txt".format(leaderClustersNum, followerClustersNum))
print(transitionMatrix.getNpTransitionMatrix())

'''
Filter booter
'''
leaderFollowerFilter = LeaderFollowerFilter(transitionMatrix)
noveltieValues = leaderFollowerFilter.getPosVelsObssNovelties(leaderPosVels
                                                              , followerPosVels)
leaderFollowerFilter.plotNovelties(noveltieValues)
print(noveltieValues)
