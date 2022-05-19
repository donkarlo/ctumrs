import numpy as np
from matplotlib import pyplot as plt

from ctumrs.TwoAlphabetWordsTransitionMatrix import TwoAlphabetWordsTransitionMatrix
from scipy.spatial.distance import directed_hausdorff

class LeaderFollowerFilter():
    def __init__(self
                 , transitionMatrix:TwoAlphabetWordsTransitionMatrix):
        self.__transitionMatrix:TwoAlphabetWordsTransitionMatrix = transitionMatrix
        self.__rangeLimit = 15000

    def getCurNoveltyValueByPrvPosVelObs(self
                                         , prvLeaderPosVelObs:tuple
                                         , prvFollowerPosVelObs:tuple
                                         , curLeaderPosVelObs:tuple
                                         , curFolowerPosVelObs:tuple)->float:

        """

        """
        leaderClusters = self.__transitionMatrix.getLeaderTimePosVelClusters()
        followerClusters = self.__transitionMatrix.getFollowerTimePosVelClusters()

        prvLeaderLabel = leaderClusters.getLabelByPosVelObs(prvLeaderPosVelObs)
        prvFollowerLabel = followerClusters.getLabelByPosVelObs(prvFollowerPosVelObs)
        prvLeaderFollowerLabelPair = (prvLeaderLabel,prvFollowerLabel)

        # Get predicted label
        predictedLabelPair = self.__transitionMatrix.getCurMostProbableLabelPairByPrvLabelPair(prvLeaderFollowerLabelPair)

        # Get predicted centers for predicted labels
        predictedLeaderCenter = leaderClusters.getClusterCenterByLabel(predictedLabelPair[0])
        predictedFollowerCenter = followerClusters.getClusterCenterByLabel(predictedLabelPair[1])

        # distance = self.__getHausedorfDistance(curLeaderPosVelObs
        #                                        ,predictedLeaderCenter
        #                                        ,curFolowerPosVelObs
        #                                        ,curFolowerPosVelObs)
        # distance = self.__getGpsBhattacharyyaDistanceAbnormalityValue(curLeaderPosVelObs
        #                                                               , predictedLeaderCenter
        #                                                               , curFolowerPosVelObs
        #                                                               , predictedFollowerCenter)

        # distance = self.__getGpsKlDistanceAbnormalityValue(curLeaderPosVelObs
        #                                                               , predictedLeaderCenter
        #                                                               , curFolowerPosVelObs
        #                                                               , predictedFollowerCenter)

        distance = self.__getGpsHellingerDistanceAbnormalityValue(curLeaderPosVelObs
                                                           , predictedLeaderCenter
                                                           , curFolowerPosVelObs
                                                           , predictedFollowerCenter)
        return distance

    def __getGpsBhattacharyyaDistanceAbnormalityValue(self
                                                      , leaderObs
                                                      , leaderPrd
                                                      , flwrObs
                                                      , flwrPrd):
        # Distribution 1
        xyCovVal = 2.0e-4
        zCovVal = 4.0e-4
        covMtx = np.array([[xyCovVal, 0, 0], [0, xyCovVal, 0], [0, 0, zCovVal]])

        leaderObsDist = {
            "mean": np.array([leaderObs[0:3]]),
            "covariance": covMtx,
        }

        leaderPrdDist = {
            "mean": np.array([leaderPrd[0:3]]),
            "covariance": covMtx,
        }

        flwrObsDist = {
            "mean": np.array([flwrObs[0:3]]),
            "covariance": covMtx,
        }

        flwrPrdDist = {
            "mean": np.array([flwrPrd[0:3]]),
            "covariance": covMtx,
        }

        leaderBhattacharyyaDistance = self.__getBhattacharyyaGaussianDistance(leaderObsDist, leaderPrdDist)
        followerBhattacharyyaDistance = self.__getBhattacharyyaGaussianDistance(flwrObsDist, flwrPrdDist)
        distance = (leaderBhattacharyyaDistance + followerBhattacharyyaDistance) / 2
        return distance

    def __getGpsKlDistanceAbnormalityValue(self
                                                      , leaderObs
                                                      , leaderPrd
                                                      , flwrObs
                                                      , flwrPrd):
        # Distribution 1
        xyCovVal = 2.0e-4
        zCovVal = 4.0e-4
        covMtx = np.array([[xyCovVal, 0, 0]
                              , [0, xyCovVal, 0]
                              , [0, 0, zCovVal]])

        leaderKlDistance = self.__getKullbackLieblerDistance(leaderObs[0:3]
                                                             , covMtx
                                                             ,leaderPrd[0:3]
                                                             , covMtx)
        followerKlDistance = self.__getKullbackLieblerDistance(flwrObs[0:3]
                                                               , covMtx
                                                               ,flwrPrd[0:3]
                                                               , covMtx)
        distance = (leaderKlDistance + followerKlDistance) / 2
        return distance

    def __getGpsHellingerDistanceAbnormalityValue(self
                                                      , leaderObs
                                                      , leaderPrd
                                                      , flwrObs
                                                      , flwrPrd):
        # Distribution 1
        xyCovVal = 2.0e-4
        zCovVal = 4.0e-4
        covMtx = np.array([[xyCovVal, 0, 0]
                              , [0, xyCovVal, 0]
                              , [0, 0, zCovVal]])

        leaderDistance = self.__getHellingerDitance(leaderObs[0:3]
                                                             , covMtx
                                                             ,leaderPrd[0:3]
                                                             , covMtx)
        followerDistance = self.__getHellingerDitance(flwrObs[0:3]
                                                               , covMtx
                                                               ,flwrPrd[0:3]
                                                               , covMtx)
        distance = (leaderDistance + followerDistance) / 2
        return distance

    def __getHausedorfDistance(self
                               ,leaderObs
                               ,leaderPrd
                               ,flwrObs
                               ,flwrPrd)->float:
        curLeaderFollowerNpPosVelObs = np.array([leaderObs, flwrObs])
        curLeaderFollowerNpPosVelCenters = np.array([leaderPrd, flwrPrd])

        hausdorfDistance1 = directed_hausdorff(curLeaderFollowerNpPosVelObs,curLeaderFollowerNpPosVelCenters)[0]
        hausdorfDistance2 = directed_hausdorff(curLeaderFollowerNpPosVelCenters,curLeaderFollowerNpPosVelObs)[0]
        distance  = max(hausdorfDistance1,hausdorfDistance2)
        return distance

    def getPosVelsObssNovelties(self
                     ,leaderPosVelObss:[tuple]
                     ,followerPosVelObss:[tuple]
                     )->np.ndarray:
        noveltyValues = []
        print("Calculating abnormality values ...")
        for leaderPosVelObsCounter,leaderPosVelObs in enumerate(leaderPosVelObss):
            if leaderPosVelObsCounter>=1 and leaderPosVelObsCounter<=self.__rangeLimit:
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
        f.set_figwidth(10)
        f.set_figheight(2.5)

        # Label
        plt.xlabel('Timestep')
        plt.ylabel('Abnormality value')
        slicedNoveltyValues = noveltyValues[0:self.__rangeLimit]
        plt.plot(range(0,self.__rangeLimit)
                 , slicedNoveltyValues
                 , label=''
                 , color='red'
                 , linewidth=1)
        # To show xlabel
        plt.tight_layout()

        # To show the inner labels
        plt.legend()

        # Novelty signal
        plt.show()


    def __getBhattacharyyaGaussianDistance(self
                                           , distribution1: dict
                                           , distribution2: dict ) -> float:
        """ Estimate Bhattacharyya Distance (between Gaussian Distributions)

        Args:
            distribution1: a sample gaussian distribution 1
            distribution2: a sample gaussian distribution 2

        Returns:
            Bhattacharyya distance
        """
        mean1 = distribution1["mean"]
        cov1 = distribution1["covariance"]

        mean2 = distribution2["mean"]
        cov2 = distribution2["covariance"]

        cov = (1 / 2) * (cov1 + cov2)

        T1 = (1 / 8) * (
            np.sqrt((mean1 - mean2) @ np.linalg.inv(cov) @ (mean1 - mean2).T)[0][0]
        )
        T2 = (1 / 2) * np.log(
            np.linalg.det(cov) / np.sqrt(np.linalg.det(cov1) * np.linalg.det(cov2))
        )

        return T1 + T2

    def __getKullbackLieblerDistance(self, m0, S0, m1, S1):
        """
        Kullback-Liebler divergence from Gaussian pm,pv to Gaussian qm,qv.
        Also computes KL divergence from a single Gaussian pm,pv to a set
        of Gaussians qm,qv.


        From wikipedia
        KL( (m0, S0) || (m1, S1))
             = .5 * ( tr(S1^{-1} S0) + log |S1|/|S0| +
                      (m1 - m0)^T S1^{-1} (m1 - m0) - N )
        """
        # store inv diag covariance of S1 and diff between means
        N = m0.shape[0]
        iS1 = np.linalg.inv(S1)
        diff = m1 - m0

        # kl is made of three terms
        tr_term = np.trace(iS1 @ S0)
        det_term = np.log(np.linalg.det(S1) / np.linalg.det(S0))  # np.sum(np.log(S1)) - np.sum(np.log(S0))
        quad_term = diff.T @ np.linalg.inv(S1) @ diff  # np.sum( (diff*diff) * iS1, axis=1)
        # print(tr_term,det_term,quad_term)
        return .5 * (tr_term + det_term + quad_term - N)

    def __getHellingerDitance(self,meanX, covX,meanY,covY):
        """ Calculates Hellinger distance between 2 multivariate normal distribution
             X = X(x1, x2)
             Y = Y(y1, y2)
             The definition can be found at https://en.wikipedia.org/wiki/Hellinger_distance
        """
        detX = np.linalg.det(covX)
        detY = np.linalg.det(covY)

        detXY = np.linalg.det((covX + covY) / 2)
        if (np.linalg.det(covX + covY) / 2) != 0:
            covXY_inverted = np.linalg.inv((covX + covY) / 2)
        else:
            covXY_inverted = np.linalg.pinv((covX + covY) / 2)
        dist = 1. - (detX ** .25 * detY ** .25 / detXY ** .5) * np.exp(
            -.125 * np.dot(np.dot(np.transpose(meanX - meanY), covXY_inverted), (meanX - meanY)))
        return min(max(dist, 0.), 1.)



