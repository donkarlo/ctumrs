import numpy as np
from sklearn.cluster import KMeans


class PosVelsClusteringStrgy:
    def __init__(self
                 , clustersNum: int
                 , posVels:np.ndarray):
        '''

        '''
        self.__clustersNum:int = clustersNum
        self.__posVels:np.ndarray = np.asarray(posVels)

        # Just for lazy loading
        self.__fittedPosVelsClusters: KMeans = None
        self.__labeledTimePosVelClustersDict = None


    def __getClusteringStrgy(self):
        return KMeans(n_clusters=self.__clustersNum , random_state=0)

    def getFittedClusters(self)->KMeans:
        '''
        Kmeans.fit return type is Kmeans,
        like this, Kmeans.labels_ and .cluster_centers_[label] etc are getting values
        '''
        if self.__fittedPosVelsClusters is None:
            print("Fitting the clusters ...")
            self.__fittedPosVelsClusters = self.__getClusteringStrgy().fit(self.__posVels)
        return self.__fittedPosVelsClusters

    def getClusterCenterByLabel(self, label:int)->tuple:
        return self.getFittedClusters().cluster_centers_[label]

    def getClustersNum(self)->int:
        return self.__clustersNum

    def getPredictedLabelByPosVelObs(self, posVel:tuple):
        posVelArr = [posVel]
        return self.getFittedClusters().predict(posVelArr)[0]

    def getLabeledPosVelsClustersDict(self) -> dict:
        """

        Returns
        -------
        object
        """

        if self.__labeledTimePosVelClustersDict is None:
            self.__labeledTimePosVelClustersDict = {}
            for curPosVel in self.__posVels:
                curPosVelLabel = self.getPredictedLabelByPosVelObs(curPosVel)
                if curPosVelLabel not in self.__labeledTimePosVelClustersDict.keys():
                    self.__labeledTimePosVelClustersDict[curPosVelLabel] = []
                self.__labeledTimePosVelClustersDict[curPosVelLabel].append(curPosVel)

        return self.__labeledTimePosVelClustersDict

    def __getPosVelDim(self):
        return self.__posVels.shape[0]


    def getClueterVelCenterByLabel(self, label):
        return self.getClusterCenterByLabel(label)[int(self.__posVels.shape[1]/2):]