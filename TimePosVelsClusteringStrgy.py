import array
from sklearn.cluster import KMeans


class TimePosVelsClusteringStrgy:
    def __init__(self
                 , clustersNum: int
                 , timePosVels
                 , posVels):
        '''

        '''
        self.__clustersNum:int = clustersNum
        self.__posVels:list = posVels
        self.__timePosVels:list = timePosVels

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
            self.__fittedPosVelsClusters = self.__getClusteringStrgy().fit(self.__posVels)
        return self.__fittedPosVelsClusters

    def getLabeledTimePosVelsClustersDict(self) -> dict:
        self.__fittedPosVelsClusters = self.getFittedClusters()

        if self.__labeledTimePosVelClustersDict is None:
            self.__labeledTimePosVelClustersDict = {}
            for labelCounter, posVelLabel in enumerate(self.__fittedPosVelsClusters.labels_):
                if posVelLabel not in self.__labeledTimePosVelClustersDict.keys():
                    self.__labeledTimePosVelClustersDict[posVelLabel] = []
                self.__labeledTimePosVelClustersDict[posVelLabel].append(self.__timePosVels[labelCounter])

        return self.__labeledTimePosVelClustersDict

    def getClusterCenterByLabel(self, label:int)->(int, int, int, int, int, int):
        return self.getFittedClusters().cluster_centers_[label]

    def getClustersNum(self)->int:
        return self.__clustersNum

    def getLabelByPosVel(self,posVel:(int,int,int,int,int,int)):
        posVelArr = [posVel]
        return self.getFittedClusters().predict(posVelArr)[0]