from ctumrs.TimePosVelObssUtility import TimePosVelObssUtility
from ctumrs.TwoAlphabetWordsTransitionMatrix import TwoAlphabetWordsTransitionMatrix
from ctumrs.TimePosVelObssPlottingUtility import TimePosVelObssPlottingUtility
from ctumrs.PosVelObssClusteringStrgy import PosVelObssClusteringStrgy

timePosVelObssPlottingUtility = TimePosVelObssPlottingUtility()
timePosVelObssUtility = TimePosVelObssUtility()

leaderClustersNum = 75
followerClustersNum = 75
velocityCoefficient = 10000

jointPathToLeaderAndFollower= "/home/donkarlo/Dropbox/projs/research/data/self-aware-drones/ctumrs/two-drones/normal-scenario/gps/"
pathToLeaderUavTimePosVelDataFile = jointPathToLeaderAndFollower+"gps-uav1-pos-vel.txt"
pathToFollowerUavTimePosVelDataFile = jointPathToLeaderAndFollower+"gps-uav2-pos-vel.txt"

leaderUavPosVelsAndTimePosVels = timePosVelObssPlottingUtility.timePosVelObssUtility(pathToLeaderUavTimePosVelDataFile, velocityCoefficient)
leaderPosVels = leaderUavPosVelsAndTimePosVels['posVels']
leaderTimePosVels = leaderUavPosVelsAndTimePosVels['timePosVels']

followerUavPosVelsAndTimePosVels = timePosVelObssPlottingUtility.timePosVelObssUtility(pathToFollowerUavTimePosVelDataFile, velocityCoefficient)
followerPosVels = followerUavPosVelsAndTimePosVels['posVels']
followerTimePosVels = followerUavPosVelsAndTimePosVels['timePosVels']

############PLOTTING THE TRAJECTORY##############
timePosVelObssPlottingUtility.plotPos(leaderPosVels)
timePosVelObssPlottingUtility.plotPos(followerPosVels)

############CLUSTERING###########
leaderTimePosVelClusteringStrgy = PosVelObssClusteringStrgy(leaderClustersNum
                                                            , leaderPosVels)
leaderTimePosVelClustersDict = leaderTimePosVelClusteringStrgy.getLabeledTimePosVelsClustersDict(leaderTimePosVels)

followerTimePosVelClusteringStrgy = PosVelObssClusteringStrgy(followerClustersNum
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