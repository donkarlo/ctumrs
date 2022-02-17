import pickle

from ctumrs.LeaderFollowerFilter import LeaderFollowerFilter
from ctumrs.TransitionMatrix import TransitionMatrix
from ctumrs.topics.rplidar.twoRpLidar.VelMul import velMul

if __name__ == "__main__":
    leaderUavClustersNum = 75
    followerUavClustersNum = 75
    velCoefficient = 10000

    jointPathToLeaderAndFollowerNormalScenario= "/home/donkarlo/Dropbox/projs/research/data/self-aware-drones/ctumrs/two-drones/normal-scenario/lidars/"
    '''Lodaing the transition matrix'''
    jointfilePathToTransitionMatrix = jointPathToLeaderAndFollowerNormalScenario + "transtionMatrix-{}*{}.txt".format(leaderUavClustersNum, followerUavClustersNum)
    transitionMatrix = TransitionMatrix()
    transitionMatrix = transitionMatrix.load(jointfilePathToTransitionMatrix)

    ''''''
    pklFile = open("/home/donkarlo/Dropbox/projs/research/data/self-aware-drones/ctumrs/two-drones/follow-scenario/lidars/"+"twoLidarsTimeRangeSumVelObss.pkl", "rb")
    leaderFollowerTimeRangeSumVelDict = pickle.load(pklFile)

    leaderPosVelObss = velMul(leaderFollowerTimeRangeSumVelDict['leaderRangeSumVelObss'],velCoefficient)
    followerPosVelObss = velMul(leaderFollowerTimeRangeSumVelDict['followerRangeSumVelObss'],velCoefficient)


    leaderFollowerFilter = LeaderFollowerFilter(transitionMatrix)
    noveltyValues = leaderFollowerFilter.getPosVelsObssNovelties(leaderPosVelObss
                                                 ,followerPosVelObss)
    leaderFollowerFilter.plotNovelties(noveltyValues)