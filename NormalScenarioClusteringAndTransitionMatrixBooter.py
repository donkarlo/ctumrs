from ctumrs.TransitionMatrix import TransitionMatrix
from ctumrs.Utility import Utility
from ctumrs.TimePosVelsClusteringStrgy import TimePosVelsClusteringStrgy

utility = Utility()
leaderClustersNum = 75
followerClustersNum = 75
velocityCoefficient = 10000

jointPathToLeaderAndFollower= "/home/donkarlo/Dropbox/projs/research/data/self-aware-drones/ctumrs/two-drones/normal-scenario/gps/"
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
leaderTimePosVelClusteringStrgy = TimePosVelsClusteringStrgy(leaderClustersNum
                                                             , leaderTimePosVels
                                                             , leaderPosVels)
leaderTimePosVelClustersDict = leaderTimePosVelClusteringStrgy.getLabeledTimePosVelsClustersDict()

followerTimePosVelClusteringStrgy = TimePosVelsClusteringStrgy(followerClustersNum
                                                               , followerTimePosVels
                                                               , followerPosVels)
followerTimePosVelClustersDict = followerTimePosVelClusteringStrgy.getLabeledTimePosVelsClustersDict()

'''PLOTING THE CLUSTERS'''
utility.plotPosWithCLusters(leaderTimePosVelClustersDict)
utility.plotPosWithCLusters(followerTimePosVelClustersDict)
utility.plotLeaderFollowerUavPosWithCLusters(leaderTimePosVelClustersDict
                                     , followerTimePosVelClustersDict)

'''
Building the transition matrix
'''
transitionMatrix = TransitionMatrix(leaderTimePosVelClusteringStrgy
                                    , followerTimePosVelClusteringStrgy
                                    , leaderTimePosVels
                                    , followerTimePosVels)
transitionMatrix.save(jointPathToLeaderAndFollower +"transtionMatrix-{}*{}.txt".format(leaderClustersNum, followerClustersNum))