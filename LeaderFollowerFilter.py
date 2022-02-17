import numpy as np
from matplotlib import pyplot as plt

from ctumrs.TransitionMatrix import TransitionMatrix
from scipy.spatial.distance import directed_hausdorff

class LeaderFollowerFilter():
    def __init__(self
                 , transitionMatrix:TransitionMatrix):
        self.__transitionMatrix:TransitionMatrix = transitionMatrix

    def getCurNoveltyValueByPrvPosVelObs(self
                                         , prvLeaderPosVelObs:tuple
                                         , prvFollowerPosVelObs:tuple
                                         , curLeaderPosVelObs:tuple
                                         , curFolowerPosVelObs:tuple)->float:

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
                     ,leaderPosVelObss:[tuple]
                     ,followerPosVelObss:[tuple]
                     )->np.ndarray:
        noveltyValues = []
        print("Calculating novelty values ...")
        for leaderPosVelObsCounter,leaderPosVelObs in enumerate(leaderPosVelObss):
            if leaderPosVelObsCounter>=1 and leaderPosVelObsCounter<50000:
                prvLeaderObs = leaderPosVelObss[leaderPosVelObsCounter-1]
                prvFolowerObs = followerPosVelObss[leaderPosVelObsCounter-1]
                curLeaderObs = leaderPosVelObs
                curFollowerObs = followerPosVelObss[leaderPosVelObsCounter]
                curNoveltyValue = self.getCurNoveltyValueByPrvPosVelObs(prvLeaderObs
                                                                             , prvFolowerObs
                                                                             , curLeaderObs
                                                                             , curFollowerObs)
                noveltyValues.append(curNoveltyValue)
        return noveltyValues

    def plotNovelties(self,noveltyValues):
        # Scale the plot
        f = plt.figure()
        f.set_figwidth(200)
        f.set_figheight(5)

        # Label
        plt.xlabel('Timestep')
        plt.ylabel('Hausdorff novelty')
        plt.plot(range(1,50000)
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

