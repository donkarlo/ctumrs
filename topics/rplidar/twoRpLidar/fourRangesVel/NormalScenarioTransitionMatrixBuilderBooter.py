import pickle

from ctumrs.PosVelObssClusteringStrgy import PosVelObssClusteringStrgy
from ctumrs.TwoAlphabetWordsTransitionMatrix import TwoAlphabetWordsTransitionMatrix
from ctumrs.topics.rplidar.twoRpLidar.fourRangesVel.TimeFourRangesVelsObss import TimeFourRangesVelsObss

sharedPathToTwoLidarsNormalScenarioFourRangesVels = "/home/donkarlo/Dropbox/projs/research/data/self-aware-drones/ctumrs/two-drones/normal-scenario/lidars/fourRangesVels/"

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
# transitionMatrix.setLeaderFollowerObsMatchStrgy("ALREADY_INDEX_MATCHED")
transitionMatrix.save(sharedPathToTwoLidarsNormalScenarioFourRangesVels + "transtionMatrix-{}*{}.txt".format(leaderClustersNum, followerClustersNum))


print("Ended")