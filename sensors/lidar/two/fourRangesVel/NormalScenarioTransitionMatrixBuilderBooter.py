import pickle

from ctumrs.TimePosVelsClusteringStrgy import TimePosVelsClusteringStrgy
from ctumrs.TransitionMatrix import TransitionMatrix
from ctumrs.sensors.lidar.two.fourRangesVel.TimeFourRangesVelsObss import TimeFourRangesVelsObss

sharedPathToTwoLidarsNormalScenarioFourRangesVels = "/home/donkarlo/Dropbox/projs/research/data/self-aware-drones/ctumrs/two-drones/normal-scenario/lidar/fourRangesVels/"

velCoefficient = 10000
leaderClustersNum = 75
followerClustersNum = 75

'''Load data'''
pklFile = open(sharedPathToTwoLidarsNormalScenarioFourRangesVels + "twoLidarsTimeFourRangesVelsObss.pkl", "rb")
leaderFollowerTimeRangeSumVelDict = pickle.load(pklFile)

leaderRangeSumVelObss, leaderTimeRangeSumVelObss = TimeFourRangesVelsObss.velMulInFourRangesVelsAndTimeFourRangesVelsObss(
    leaderFollowerTimeRangeSumVelDict["leaderFourRangesVelsObss"]
    ,leaderFollowerTimeRangeSumVelDict["leaderTimeFourRangesVelsObss"]
    , velCoefficient)

followerRangeSumVelObss, followerTimeRangeSumVelObss = TimeFourRangesVelsObss.velMulInFourRangesVelsAndTimeFourRangesVelsObss(
    leaderFollowerTimeRangeSumVelDict["followerFourRangesVelsObss"]
    ,leaderFollowerTimeRangeSumVelDict["followerTimeFourRangesVelsObss"]
    ,velCoefficient)


'''Cluster each'''

leaderTimeRangeSumVelClusteringStrgy = TimePosVelsClusteringStrgy(leaderClustersNum
                                                                  , leaderTimeRangeSumVelObss
                                                                  , leaderRangeSumVelObss)

followerTimePosVelClusteringStrgy = TimePosVelsClusteringStrgy(followerClustersNum
                                                               , followerTimeRangeSumVelObss
                                                               , followerRangeSumVelObss)
'''Build transition matrix'''
transitionMatrix = TransitionMatrix(leaderTimeRangeSumVelClusteringStrgy
                                    , followerTimePosVelClusteringStrgy
                                    , leaderTimeRangeSumVelObss
                                    , followerTimeRangeSumVelObss)
# transitionMatrix.setLeaderFollowerObsMatchStrgy("ALREADY_INDEX_MATCHED")
transitionMatrix.save(sharedPathToTwoLidarsNormalScenarioFourRangesVels + "transtionMatrix-{}*{}.txt".format(leaderClustersNum, followerClustersNum))


print("Ended")