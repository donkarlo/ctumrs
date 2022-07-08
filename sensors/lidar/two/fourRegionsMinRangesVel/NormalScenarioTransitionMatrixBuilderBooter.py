import pickle

from ctumrs.TimePosVelsClusteringStrgy import TimePosVelsClusteringStrgy
from ctumrs.TransitionMatrix import TransitionMatrix
from ctumrs.sensors.lidar.two.fourRegionsMinRangesVel.TimeFourRegionsMinVelsObss import \
    TimeFourRegionsMinRangesVelsObss

sharedPathToTwoLidarsNormalScenarioFourRangesVels = "/home/donkarlo/Dropbox/projs/research/data/self-aware-drones/ctumrs/two-drones/normal-scenario/lidar/fourRegionsMinRangesVels/"

velCoefficient = 10000
leaderClustersNum = 75
followerClustersNum = 75

'''Load data'''
pklFile = open(sharedPathToTwoLidarsNormalScenarioFourRangesVels + "twoLidarsTimeFourRegionsMinRangesVelsObss.pkl", "rb")
leaderFollowerTimeRangeSumVelDict = pickle.load(pklFile)

leaderRangeSumVelObss, leaderTimeRangeSumVelObss = TimeFourRegionsMinRangesVelsObss.velMulInFourRangesVelsAndTimeFourRangesVelsObss(
    leaderFollowerTimeRangeSumVelDict["leaderFourRegionsMinRangesVelsObss"]
    ,leaderFollowerTimeRangeSumVelDict["leaderTimeFourRegionsMinRangesVelsObss"]
    , velCoefficient)

followerRangeSumVelObss, followerTimeRangeSumVelObss = TimeFourRegionsMinRangesVelsObss.velMulInFourRangesVelsAndTimeFourRangesVelsObss(
    leaderFollowerTimeRangeSumVelDict["followerFourRegionsMinRangesVelsObss"]
    ,leaderFollowerTimeRangeSumVelDict["followerTimeFourRegionsMinRangesVelsObss"]
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


print("Transition matrix built")