import pickle

import numpy as np

from ctumrs.TimePosVelsClusteringStrgy import TimePosVelsClusteringStrgy


class OneAlphabetWordsTransitionMatrix:
    def __init__(self
                 ,posVelObssClusteringStrgy:TimePosVelsClusteringStrgy
                 ,posVelObss:np.array):
        self.__posVelObssClusteringStrgy = posVelObssClusteringStrgy
        self.__posVelObss = posVelObss
        #lazy loading
        self.__npTransitionMatrix = None

    def getNpTransitionMatrix(self) -> np.array:
        """"""
        if self.__npTransitionMatrix is None:
            print ("Building one alphabet words transition matrix ...")
            self.__npTransitionMatrix = np.zeros((self.__posVelObssClusteringStrgy.getClustersNum()
                                                  ,self.__posVelObssClusteringStrgy.getClustersNum()))
            for posVelObssCounter,curPosVelObs in enumerate(self.__posVelObss):
                if posVelObssCounter < 100000:
                    if posVelObssCounter > 1:
                        prvLabel = self.__posVelObssClusteringStrgy.getPredictedLabelByPosVelObs(self.__posVelObss[posVelObssCounter - 1])
                        curLabel = self.__posVelObssClusteringStrgy.getPredictedLabelByPosVelObs(self.__posVelObss[posVelObssCounter])
                        self.__npTransitionMatrix[prvLabel][curLabel]+=1

        return self.__npTransitionMatrix

    def getHighestPorobabelNextLabelBasedOnthePrvOne(self,prvLabel:int)->int:
        return np.argmax(self.getNpTransitionMatrix()[prvLabel])


    def save(self,filePath)->None:
        """"""
        self.getNpTransitionMatrix()
        with open(filePath, 'wb') as file:
            pickle.dump(self, file)

    def load(self,filePath)->pickle:
        """"""
        with open(filePath, 'rb') as file:
            loadedPickle = pickle.load(file)
            return loadedPickle

    def getClusteringStrgy(self)->TimePosVelsClusteringStrgy:
        return self.__posVelObssClusteringStrgy