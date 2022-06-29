import pickle

import numpy as np

from ctumrs.TimePosVelsClusteringStrgy import TimePosVelsClusteringStrgy
from ctumrs.TransitionMatrix import TransitionMatrix
from ctumrs.topics.rplidar.twoRpLidar.ranges.TimeRangesVelsObss import TimeRangesVelsObss
from MachineSettings import MachineSettings

sharedPathToTwoLidarsNormalScenarioFourRangesVels = "{}projs/research/data/self-aware-drones/ctumrs/two-drones/normal-scenario/lidars/allRanges/".format(MachineSettings.MAIN_PATH)

velCoefficient = 20
leaderClustersNum = 75
followerClustersNum = 75

'''Load data'''
pklFile = open(sharedPathToTwoLidarsNormalScenarioFourRangesVels + "twoLidarsTimeRangesVelsObss.pkl", "rb")
leaderFollowerTimeRangeSumVelDict = pickle.load(pklFile)

npLeaderTimeRangesVelsObss = np.array(leaderFollowerTimeRangeSumVelDict["leaderTimeRangesVelsObss"])
npFollowerTimeRangesVelsObss = np.array(leaderFollowerTimeRangeSumVelDict["followerTimeRangesVelsObss"])

npLeaderTimeRangesCoefVelsObss = TimeRangesVelsObss.velMulInTimeRangesVelsObss(npLeaderTimeRangesVelsObss
                                                                               , velCoefficient)
npLeaderRangesCoefVelsObss = npLeaderTimeRangesCoefVelsObss[:,1:]
npFollowerTimeRangesCoefVelsObss = TimeRangesVelsObss.velMulInTimeRangesVelsObss(npFollowerTimeRangesVelsObss
                                                                               , velCoefficient)
npFollowerRangesCoefVelsObss = npFollowerTimeRangesCoefVelsObss[:,1:]

'''Cluster each'''

leaderTimeRangeSumVelClusteringStrgy = TimePosVelsClusteringStrgy(leaderClustersNum
                                                                  , npLeaderRangesCoefVelsObss)

followerTimePosVelClusteringStrgy = TimePosVelsClusteringStrgy(followerClustersNum
                                                               , npFollowerRangesCoefVelsObss)
'''Build transition matrix'''
transitionMatrix = TransitionMatrix(leaderTimeRangeSumVelClusteringStrgy
                                    , followerTimePosVelClusteringStrgy
                                    , npLeaderTimeRangesCoefVelsObss
                                    , npFollowerTimeRangesCoefVelsObss)
# transitionMatrix.setLeaderFollowerObsMatchStrgy("ALREADY_INDEX_MATCHED")
transitionMatrix.save(sharedPathToTwoLidarsNormalScenarioFourRangesVels + "transtionMatrix-clusters-{}-{}-velco-{}.txt".format(leaderClustersNum
                                                                                                                               , followerClustersNum
                                                                                                                               , velCoefficient
                                                                                                                               ))


print("Ended")