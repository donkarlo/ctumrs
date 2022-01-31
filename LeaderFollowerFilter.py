import numpy as np
from matplotlib import pyplot as plt

from ctumrs.TransitionMatrix import TransitionMatrix
from scipy.spatial.distance import directed_hausdorff

class LeaderFollowerFilter():
    def __init__(self
                 , transitionMatrix:TransitionMatrix):
        self.__transitionMatrix:TransitionMatrix = transitionMatrix

    def getCurNoveltyValueByPrvPosVelObs(self
                                         , prvLeaderPosVelObs:(int, int, int, int, int, int)
                                         , prvFollowerPosVelObs:(int, int, int, int, int, int)
                                         , curLeaderPosVelObs:(int, int, int, int, int, int)
                                         , curFolowerPosVelObs:(int, int, int, int, int, int))->float:

        """

        """
        leaderClusters = self.__transitionMatrix.getLeaderTimePosVelClusters()
        followerClusters = self.__transitionMatrix.getFollowerTimePosVelClusters()

        prvLeaderLabel = leaderClusters.getLabelByPosVel(prvLeaderPosVelObs)
        prvFollowerLabel = followerClusters.getLabelByPosVel(prvFollowerPosVelObs)
        prvLeaderFollowerLabelPair = (prvLeaderLabel,prvFollowerLabel)

        # Get predicted label
        predictedLabelPair = self.__transitionMatrix.getCurMostProbableLabelPairByPrvLabelPair(prvLeaderFollowerLabelPair)

        # Get predicted centers for predicted labels
        predictedLeaderCenter = leaderClusters.getClusterCenterByLabel(predictedLabelPair[0])
        predictedFollowerCenter = followerClusters.getClusterCenterByLabel(predictedLabelPair[1])

        curLeaderFollowerNpPosVelObs = np.array([curLeaderPosVelObs, curFolowerPosVelObs])
        curLeaderFollowerNpPosVelCenters = np.array([predictedLeaderCenter,predictedFollowerCenter])

        hausdorfDistance1 = directed_hausdorff(curLeaderFollowerNpPosVelObs,curLeaderFollowerNpPosVelCenters)[0]
        hausdorfDistance2 = directed_hausdorff(curLeaderFollowerNpPosVelCenters,curLeaderFollowerNpPosVelObs)[0]

        return max(hausdorfDistance1,hausdorfDistance2)

    def getPosVelsObssNovelties(self
                     ,leaderPosVelObss:[(int,int,int,int,int,int)]
                     ,followerPosVelObss:[(int,int,int,int,int,int)]
                     )->np.ndarray:
        noveltyValues = []
        for leaderPosVelObsCounter,leaderPosVelObs in enumerate(leaderPosVelObss):
            if leaderPosVelObsCounter>=1 and leaderPosVelObsCounter<60000:
                prvLeaderObs = leaderPosVelObss[leaderPosVelObsCounter-1]
                prvFolowerObs = followerPosVelObss[leaderPosVelObsCounter-1]
                curLeaderObs = leaderPosVelObs
                curFollowerObs = followerPosVelObss[leaderPosVelObsCounter]
                curNoveltyValue = self.getCurNoveltyValueByPrvPosVelObs(prvLeaderObs
                                                                             , prvFolowerObs
                                                                             , curLeaderObs
                                                                             , curFollowerObs)
                print(curNoveltyValue)
                noveltyValues.append(curNoveltyValue)
        return noveltyValues

    def plotNovelties(self,noveltyValues):
        # Scale the plot
        f = plt.figure()
        f.set_figwidth(100)
        f.set_figheight(1)

        # Label
        plt.xlabel('Timestep')
        plt.ylabel('Hausdorff novelty')
        plt.plot(range(1,60000)
                 , noveltyValues
                 , label='Novelty'
                 , color='red'
                 , linewidth=1)
        # To show xlabel
        plt.tight_layout()

        # To show the inner labels
        plt.legend()

        # Novelty signal
        plt.show()

