from sklearn.cluster import KMeans


class TimePosVelsClusteringStrgy:
    def __init__(self
                 , clustersNum: int
                 , posVelObss):
        '''

        '''
        self.__clustersNum:int = clustersNum
        self.__posVelObss:list = posVelObss

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
            self.__fittedPosVelsClusters = self.__getClusteringStrgy().fit(self.__posVelObss)
        return self.__fittedPosVelsClusters

    def getClusterCenterByLabel(self, label:int)->tuple:
        return self.getFittedClusters().cluster_centers_[label]

    def getClustersNum(self)->int:
        return self.__clustersNum

    def getLabelByPosVelObs(self, posVel:tuple):
        posVelArr = [posVel]
        return self.getFittedClusters().predict(posVelArr)[0]

    def getLabeledTimePosVelsClustersDict(self,timePosVels) -> dict:
        self.__fittedPosVelsClusters = self.getFittedClusters()

        if self.__labeledTimePosVelClustersDict is None:
            self.__labeledTimePosVelClustersDict = {}
            for labelCounter, posVelLabel in enumerate(self.__fittedPosVelsClusters.labels_):
                if posVelLabel not in self.__labeledTimePosVelClustersDict.keys():
                    self.__labeledTimePosVelClustersDict[posVelLabel] = []
                self.__labeledTimePosVelClustersDict[posVelLabel].append(timePosVels[labelCounter])

        return self.__labeledTimePosVelClustersDict