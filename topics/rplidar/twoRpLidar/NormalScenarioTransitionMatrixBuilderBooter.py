import pickle

from ctumrs.TimePosVelsClusteringStrgy import TimePosVelsClusteringStrgy
from ctumrs.TransitionMatrix import TransitionMatrix

dataPathToLidarOfTwoDronesTopic = "/home/donkarlo/Dropbox/projs/research/data/self-aware-drones/ctumrs/two-drones/normal-scenario/lidars/"

velCoefficient = 10000
leaderClustersNum = 75
followerClustersNum = 75
'''Load data'''
pklFile = open(dataPathToLidarOfTwoDronesTopic+"twoLidarsTimeRangeSumVelObss.pkl", "rb")
leaderFollowerTimeRangeSumVelDict = pickle.load(pklFile)

def velMul(posVelObss:list, timePosVelObss:list, velCoefficient:float):
    """"""
    for counter,element in enumerate(posVelObss):
        posVelObss[counter][1] = velCoefficient * posVelObss[counter][1]
        timePosVelObss[counter][2] = velCoefficient * timePosVelObss[counter][2]
    return [posVelObss, timePosVelObss]


leaderPosVels,leaderTimePosVels = velMul(
    leaderFollowerTimeRangeSumVelDict["leaderRangeSumVelObss"]
    ,leaderFollowerTimeRangeSumVelDict["leaderTimeRangeSumVelObss"]
    , velCoefficient)

followerPosVels,followerTimePosVels = velMul(
    leaderFollowerTimeRangeSumVelDict["followerRangeSumVelObss"]
    ,leaderFollowerTimeRangeSumVelDict["followerTimeRangeSumVelObss"]
    ,velCoefficient)


'''Cluster each'''

leaderTimePosVelClusteringStrgy = TimePosVelsClusteringStrgy(leaderClustersNum
                                                             , leaderTimePosVels
                                                             , leaderPosVels)
leaderTimePosVelClustersDict = leaderTimePosVelClusteringStrgy.getLabeledTimePosVelsClustersDict()

followerTimePosVelClusteringStrgy = TimePosVelsClusteringStrgy(followerClustersNum
                                                               , followerTimePosVels
                                                               , followerPosVels)
followerTimePosVelClustersDict = followerTimePosVelClusteringStrgy.getLabeledTimePosVelsClustersDict()



'''Build transition matrix'''
transitionMatrix = TransitionMatrix(leaderTimePosVelClusteringStrgy
                                    , followerTimePosVelClusteringStrgy
                                    , leaderTimePosVels
                                    , followerTimePosVels)
# transitionMatrix.setLeaderFollowerObsMatchStrgy("ALREADY_INDEX_MATCHED")
transitionMatrix.save(dataPathToLidarOfTwoDronesTopic +"transtionMatrix-{}*{}.txt".format(leaderClustersNum, followerClustersNum))


print("Ended")