import pickle

from ctumrs.TimePosVelsClusteringStrgy import TimePosVelsClusteringStrgy
from ctumrs.TransitionMatrix import TransitionMatrix

dataPathToLidarOfTwoDronesTopic = "/home/donkarlo/Dropbox/projs/research/data/self-aware-drones/ctumrs/two-drones/inner-squares/lidar/"

velCoefficient = 10000
'''Load data'''
pklFile = open(dataPathToLidarOfTwoDronesTopic+"twoLidarsTimeRangeSumVelsObss.pkl", "rb")
leaderFollowerTimeRangeSumVelDict = pickle.load(pklFile)

def velMul(posVelObss:list, timePosVelObss:list, velCoefficient:float):
    for counter,element in enumerate(posVelObss):
        posVelObss[counter][1] = velCoefficient * posVelObss[counter][1]
        timePosVelObss[counter][2] = velCoefficient * timePosVelObss[counter][2]
    return [posVelObss, timePosVelObss]


leaderPosVels,leaderTimePosVels = velMul(
    leaderFollowerTimeRangeSumVelDict["leaderRangeSumRangeVelObss"]
    ,leaderFollowerTimeRangeSumVelDict["leaderTimeRangeSumRangeVelObss"]
    , velCoefficient)

followerPosVels,followerTimePosVels = velMul(
    leaderFollowerTimeRangeSumVelDict["followerRangeSumRangeVelObss"]
    ,leaderFollowerTimeRangeSumVelDict["followerTimeRangeSumRangeVelObss"]
    ,velCoefficient)


'''Cluster each'''
leaderClustersNum = 75
followerClustersNum = 75

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
transitionMatrix.save(dataPathToLidarOfTwoDronesTopic +"transtionMatrix-{}*{}.txt".format(leaderClustersNum, followerClustersNum))


print("Ended")