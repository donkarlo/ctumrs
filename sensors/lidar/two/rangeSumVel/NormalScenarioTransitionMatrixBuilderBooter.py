import pickle

from ctumrs.PosVelsClusteringStrgy import PosVelsClusteringStrgy
from ctumrs.TwoAlphabetWordsTransitionMatrix import TwoAlphabetWordsTransitionMatrix
from ctumrs.sensors.lidar.two.rangeSumVel.TimeRangeSumVelObss import TimeRangeSumVelObss

dataPathToLidarOfTwoDronesTopic = "/home/donkarlo/Dropbox/projs/research/data/self-aware-drones/ctumrs/two-drones/normal-scenario/lidar/rangeSumVel/"

velCoefficient = 10000
leaderClustersNum = 75
followerClustersNum = 75

'''Load data'''
pklFile = open(dataPathToLidarOfTwoDronesTopic+"twoLidarsTimeRangeSumVelObss.pkl", "rb")
leaderFollowerTimeRangeSumVelDict = pickle.load(pklFile)

leaderRangeSumVelObss, leaderTimeRangeSumVelObss = TimeRangeSumVelObss.velMulInRangeSumAndTimeRangeSumObss(
    leaderFollowerTimeRangeSumVelDict["leaderRangeSumVelObss"]
    ,leaderFollowerTimeRangeSumVelDict["leaderTimeRangeSumVelObss"]
    , velCoefficient)

followerRangeSumVelObss, followerTimeRangeSumVelObss = TimeRangeSumVelObss.velMulInRangeSumAndTimeRangeSumObss(
    leaderFollowerTimeRangeSumVelDict["followerRangeSumVelObss"]
    ,leaderFollowerTimeRangeSumVelDict["followerTimeRangeSumVelObss"]
    ,velCoefficient)


'''Cluster each'''

leaderTimeRangeSumVelClusteringStrgy = PosVelsClusteringStrgy(leaderClustersNum
                                                              , leaderTimeRangeSumVelObss
                                                              , leaderRangeSumVelObss)

followerTimePosVelClusteringStrgy = PosVelsClusteringStrgy(followerClustersNum
                                                           , followerTimeRangeSumVelObss
                                                           , followerRangeSumVelObss)
'''Build transition matrix'''
transitionMatrix = TwoAlphabetWordsTransitionMatrix(leaderTimeRangeSumVelClusteringStrgy
                                                    , followerTimePosVelClusteringStrgy
                                                    , leaderTimeRangeSumVelObss
                                                    , followerTimeRangeSumVelObss)
transitionMatrix.save(dataPathToLidarOfTwoDronesTopic +"transtionMatrix-{}*{}.txt".format(leaderClustersNum, followerClustersNum))


print("Ended")