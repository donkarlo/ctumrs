import pickle

from ctumrs.PosVelObssClusteringStrgy import PosVelObssClusteringStrgy
from ctumrs.TwoAlphabetWordsTransitionMatrix import TwoAlphabetWordsTransitionMatrix
from ctumrs.topics.rplidar.twoRpLidar.rangeSumVel.TimeRangeSumVelObss import TimeRangeSumVelObss

dataPathToLidarOfTwoDronesTopic = "/home/donkarlo/Dropbox/projs/research/data/self-aware-drones/ctumrs/two-drones/normal-scenario/lidars/rangeSumVel/"

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

leaderTimeRangeSumVelClusteringStrgy = PosVelObssClusteringStrgy(leaderClustersNum
                                                                 , leaderTimeRangeSumVelObss
                                                                 , leaderRangeSumVelObss)

followerTimePosVelClusteringStrgy = PosVelObssClusteringStrgy(followerClustersNum
                                                              , followerTimeRangeSumVelObss
                                                              , followerRangeSumVelObss)
'''Build transition matrix'''
transitionMatrix = TwoAlphabetWordsTransitionMatrix(leaderTimeRangeSumVelClusteringStrgy
                                                    , followerTimePosVelClusteringStrgy
                                                    , leaderTimeRangeSumVelObss
                                                    , followerTimeRangeSumVelObss)
transitionMatrix.save(dataPathToLidarOfTwoDronesTopic +"transtionMatrix-{}*{}.txt".format(leaderClustersNum, followerClustersNum))


print("Ended")