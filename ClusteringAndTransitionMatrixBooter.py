from ctumrs.TransitionMatrix import TransitionMatrix
from ctumrs.Utility import Utility
from ctumrs.TimePosVelsClusteringStrgy import TimePosVelsClusteringStrgy
from ctumrs.LeaderFollowerFilter import LeaderFollowerFilter
utility = Utility()
leaderClustersNum = 100
followerClustersNum = 100
velocityCoefficient = 10000

jointPathToLeaderAndFollower= "/home/donkarlo/Dropbox/projs/research/data/self-aware-drones/ctumrs/two-drones/inner-squares/"
pathToLeaderUavTimePosVelDataFile = jointPathToLeaderAndFollower+"gps-uav1-pos-vel.txt"
pathToFollowerUavTimePosVelDataFile = jointPathToLeaderAndFollower+"gps-uav2-pos-vel.txt"

leaderUavPosVelsAndTimePosVels = utility.getTimePosVelsAndPosVels(pathToLeaderUavTimePosVelDataFile, velocityCoefficient)
leaderPosVels = leaderUavPosVelsAndTimePosVels['posVels']
leaderTimePosVels = leaderUavPosVelsAndTimePosVels['timePosVels']

followerUavPosVelsAndTimePosVels = utility.getTimePosVelsAndPosVels(pathToFollowerUavTimePosVelDataFile, velocityCoefficient)
followerPosVels = followerUavPosVelsAndTimePosVels['posVels']
followerTimePosVels = followerUavPosVelsAndTimePosVels['timePosVels']

############PLOTTING THE TRAJECTORY##############
utility.plotPos(leaderPosVels)
utility.plotPos(followerPosVels)

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
utility.plotPosWithCLusters(leaderTimePosVelClustersDict)
utility.plotPosWithCLusters(followerTimePosVelClustersDict)
utility.plotLeaderFollowerUavPosWithCLusters(leaderTimePosVelClustersDict
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
Filter booter: Abnormal scenario
'''
jointPathToLeaderAndFollower= "/home/donkarlo/Dropbox/projs/research/data/self-aware-drones/ctumrs/two-drones/inner-squares-decrease-distance-2-p-7/"
pathToLeaderUavTimePosVelDataFile = jointPathToLeaderAndFollower+"uav-1-cleaned-gps-pos-vel.txt"
pathToFollowerUavTimePosVelDataFile = jointPathToLeaderAndFollower+"uav-2-cleaned-gps-pos-vel.txt"

leaderUavPosVelsAndTimePosVels = utility.getTimePosVelsAndPosVels(pathToLeaderUavTimePosVelDataFile, velocityCoefficient)
leaderPosVels = leaderUavPosVelsAndTimePosVels['posVels']
leaderTimePosVels = leaderUavPosVelsAndTimePosVels['timePosVels']

followerUavPosVelsAndTimePosVels = utility.getTimePosVelsAndPosVels(pathToFollowerUavTimePosVelDataFile, velocityCoefficient)
followerPosVels = followerUavPosVelsAndTimePosVels['posVels']
followerTimePosVels = followerUavPosVelsAndTimePosVels['timePosVels']



leaderFollowerFilter = LeaderFollowerFilter(transitionMatrix)
noveltieValues = leaderFollowerFilter.getPosVelsObssNovelties(leaderPosVels
                                                              , followerPosVels)
leaderFollowerFilter.plotNovelties(noveltieValues)
print(noveltieValues)
