from ctumrs.TimePosVelObssUtility import TimePosVelObssUtility
from ctumrs.TwoAlphabetWordsTransitionMatrix import TwoAlphabetWordsTransitionMatrix
from ctumrs.TimePosVelObssPlottingUtility import TimePosVelObssPlottingUtility
from ctumrs.TimePosVelsClusteringStrgy import TimePosVelsClusteringStrgy

timePosVelObssPlottingUtility = TimePosVelObssPlottingUtility()
timePosVelObssUtility = TimePosVelObssUtility()

leaderClustersNum = 75
followerClustersNum = 75
velocityCoefficient = 10000

jointPathToLeaderAndFollower= "/home/donkarlo/Dropbox/projs/research/data/self-aware-drones/ctumrs/two-drones/normal-scenario/gps/"
pathToLeaderUavTimePosVelDataFile = jointPathToLeaderAndFollower+"uav-1-cleaned-gps-pos-vel.txt"
pathToFollowerUavTimePosVelDataFile = jointPathToLeaderAndFollower+"uav-2-cleaned-gps-pos-vel.txt"

leaderUavPosVelsAndTimePosVels = timePosVelObssUtility.getTimePosVelsAndPosVels(pathToLeaderUavTimePosVelDataFile, velocityCoefficient)
leaderPosVels = leaderUavPosVelsAndTimePosVels['posVels']
leaderTimePosVels = leaderUavPosVelsAndTimePosVels['timePosVels']

followerUavPosVelsAndTimePosVels = timePosVelObssUtility.getTimePosVelsAndPosVels(pathToFollowerUavTimePosVelDataFile, velocityCoefficient)
followerPosVels = followerUavPosVelsAndTimePosVels['posVels']
followerTimePosVels = followerUavPosVelsAndTimePosVels['timePosVels']

############PLOTTING THE TRAJECTORY##############
timePosVelObssPlottingUtility.plotPos(leaderPosVels)
timePosVelObssPlottingUtility.plotPos(followerPosVels)

############CLUSTERING###########
leaderTimePosVelClusteringStrgy = TimePosVelsClusteringStrgy(leaderClustersNum
                                                             , leaderPosVels)
leaderTimePosVelClustersDict = leaderTimePosVelClusteringStrgy.getLabeledTimePosVelsClustersDict(leaderTimePosVels)

followerTimePosVelClusteringStrgy = TimePosVelsClusteringStrgy(followerClustersNum
                                                               , followerPosVels)
followerTimePosVelClustersDict = followerTimePosVelClusteringStrgy.getLabeledTimePosVelsClustersDict(followerTimePosVels)

'''PLOTING THE CLUSTERS'''
timePosVelObssPlottingUtility.plotPosWithCLusters(leaderTimePosVelClustersDict)
timePosVelObssPlottingUtility.plotPosWithCLusters(followerTimePosVelClustersDict)
timePosVelObssPlottingUtility.plotLeaderFollowerUavPosWithCLusters(leaderTimePosVelClustersDict
                                                                   , followerTimePosVelClustersDict)

'''
Building the transition matrix
'''
transitionMatrix = TwoAlphabetWordsTransitionMatrix(leaderTimePosVelClusteringStrgy
                                                    , followerTimePosVelClusteringStrgy
                                                    , leaderTimePosVels
                                                    , followerTimePosVels)
transitionMatrix.save(jointPathToLeaderAndFollower +"transtionMatrix-{}*{}.txt".format(leaderClustersNum, followerClustersNum))