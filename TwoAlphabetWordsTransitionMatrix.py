import numpy as np
from ctumrs.TimePosVelsClusteringStrgy import TimePosVelsClusteringStrgy
import pickle

from ctumrs.TimePosVelObssUtility import TimePosVelObssUtility


class TwoAlphabetWordsTransitionMatrix:
    def __init__(self
                 , leaderTimePosVelClusters:TimePosVelsClusteringStrgy = None
                 , followerTimePosVelClusters:TimePosVelsClusteringStrgy = None
                 , leaderTimePosVels:list = None
                 , followerTimePosVels:list = None
                 ):

        self.__robot1TimePosVelClusters: TimePosVelsClusteringStrgy = leaderTimePosVelClusters
        self.__robot2TimePosVelClusters: TimePosVelsClusteringStrgy = followerTimePosVelClusters

        self.__robot1TimePosVels = leaderTimePosVels
        self.__robot2TimePosVels = followerTimePosVels

        self.__npTransitionMatrix = None
        self.__mappingRowOrColNumLabelMapDict = None

        self.__robot1And2ObsMatchStrgy = "FIND_BY_TIMESTAMP"

    def setLeaderFollowerObsMatchStrgy(self,leaderFollowerObsMatchStrgy="ALREADY_INDEX_MATCHED"):
        self.__robot1And2ObsMatchStrgy = leaderFollowerObsMatchStrgy

    def getRobot1TimePosVelClusters(self)->TimePosVelsClusteringStrgy:
        return self.__robot1TimePosVelClusters

    def getRobot2TimePosVelClusters(self)->TimePosVelsClusteringStrgy:
        return self.__robot2TimePosVelClusters

    def __getRowAndColByTwoLabelPairs(self, prvCurFollowerLeaderLabels: ((int, int)
                                                                         , (int, int)))->(int, int):
        rowNum = self.getTransitionMatrixRowOrColIdxByLabelPair(prvCurFollowerLeaderLabels[0])
        colNum = self.getTransitionMatrixRowOrColIdxByLabelPair(prvCurFollowerLeaderLabels[1])
        return (rowNum,colNum)

    def __increaseComponentOfTwoLabelPairsByOne(self, prvCurFollowerLeaderLabels: ((int, int), (int, int)))->None:
        rowNum,colNum = self.__getRowAndColByTwoLabelPairs(prvCurFollowerLeaderLabels)
        # print (prvCurFollowerLeaderLabels, (rowNum, colNum))
        self.__npTransitionMatrix[rowNum][colNum] += 1

    def __getMatrixValForLabelPairs(self, prvCurFollowerLeaderLabels: ((int, int), (int, int))) -> float:
        rowNum = self.getTransitionMatrixRowOrColIdxByLabelPair(prvCurFollowerLeaderLabels[0])
        colNum = self.getTransitionMatrixRowOrColIdxByLabelPair(prvCurFollowerLeaderLabels[1])
        return self.__npTransitionMatrix[rowNum][colNum]

    def __getMappingRowOrColNumLabelMapDict(self)->dict:
        if self.__mappingRowOrColNumLabelMapDict is None:
            counter = 0
            self.__mappingRowOrColNumLabelMapDict = {}
            for i in range(0, self.__robot1TimePosVelClusters.getClustersNum()):
                for j in range(0, self.__robot2TimePosVelClusters.getClustersNum()):
                    self.__mappingRowOrColNumLabelMapDict[counter] = (i, j)
                    counter += 1
        return self.__mappingRowOrColNumLabelMapDict

    def __getLabelPairByIdx(self, idx:int)->(int,int):
        for curIdx in self.__getMappingRowOrColNumLabelMapDict():
            if idx == curIdx:
                labelPair = self.__getMappingRowOrColNumLabelMapDict()[idx]
                return labelPair

    def getTransitionMatrixRowOrColIdxByLabelPair(self, labelPair: (int, int))->int:
        for rowOrColNum in self.__getMappingRowOrColNumLabelMapDict():
            if self.__mappingRowOrColNumLabelMapDict[rowOrColNum][0] == labelPair[0] and self.__mappingRowOrColNumLabelMapDict[rowOrColNum][1] == labelPair[1]:
                return rowOrColNum

    def getNpTransitionMatrix(self)->np.array:
        if self.__npTransitionMatrix is None:
            print ("Building two alphabet words transition matrix ...")
            self.__npTransitionMatrix = np.zeros((self.__robot1TimePosVelClusters.getClustersNum() * self.__robot2TimePosVelClusters.getClustersNum()
             , self.__robot1TimePosVelClusters.getClustersNum() * self.__robot2TimePosVelClusters.getClustersNum()))
            foundFollowerLabels = []
            foundLeaderLabels = []
            for leaderUavCurTimePosVelCounter, curLeaderUavTimePosVel in enumerate(self.__robot1TimePosVels):
                if leaderUavCurTimePosVelCounter < 100000:
                    if leaderUavCurTimePosVelCounter > 1:
                        curClosestFollowerTimePosVel = TimePosVelObssUtility.findClosestTimeWiseFollowerTimePosVelToLeaderTimePosVel(
                            curLeaderUavTimePosVel
                        , self.__robot2TimePosVels) if self.__robot1And2ObsMatchStrgy != "ALREADY_INDEX_MATCHED" else self.__robot2TimePosVels[leaderUavCurTimePosVelCounter]

                        # find previous leader follower timewisely
                        prvLeaderUavTimePosVel = self.__robot1TimePosVels[leaderUavCurTimePosVelCounter - 1]
                        prvClosestFollowerTimePosVel = TimePosVelObssUtility.findClosestTimeWiseFollowerTimePosVelToLeaderTimePosVel(
                            prvLeaderUavTimePosVel
                            ,self.__robot2TimePosVels)  if self.__robot1And2ObsMatchStrgy != "ALREADY_INDEX_MATCHED" else self.__robot2TimePosVels[leaderUavCurTimePosVelCounter - 1]

                        # Finding the label for previous leader follower  observation
                        prvLeaderUavLabel = \
                        self.__robot1TimePosVelClusters.getFittedClusters().predict([TimePosVelObssUtility.getPosVelByTimePosVel(prvLeaderUavTimePosVel)])[0]
                        prvFollowerUavLabel = self.__robot2TimePosVelClusters.getFittedClusters().predict(
                            [TimePosVelObssUtility.getPosVelByTimePosVel(prvClosestFollowerTimePosVel)])[0]

                        # Finding the label for current leader follower  observation
                        curLeaderUavLabel = \
                        self.__robot1TimePosVelClusters.getFittedClusters().predict([TimePosVelObssUtility.getPosVelByTimePosVel(curLeaderUavTimePosVel)])[0]
                        curFollowerUavLabel = self.__robot2TimePosVelClusters.getFittedClusters().predict(
                            [TimePosVelObssUtility.getPosVelByTimePosVel(curClosestFollowerTimePosVel)])[0]

                        if curFollowerUavLabel not in foundFollowerLabels:
                            foundFollowerLabels.append(curFollowerUavLabel)
                        if curLeaderUavLabel not in foundLeaderLabels:
                            foundLeaderLabels.append(curLeaderUavLabel)

                        # Increasing the transition matrix
                        self.__increaseComponentOfTwoLabelPairsByOne(((prvLeaderUavLabel, prvFollowerUavLabel)
                                                                                , (curLeaderUavLabel, curFollowerUavLabel)))
        return self.__npTransitionMatrix

    def getCurMostProbableLabelPairByPrvLabelPair(self
                            ,prvLabelPair:(int,int))->(int,int):
        rowNum = self.getTransitionMatrixRowOrColIdxByLabelPair(prvLabelPair)
        maxIdx = np.argmax(self.__npTransitionMatrix[rowNum])
        return self.__getLabelPairByIdx(maxIdx)

    def save(self,filePath)->None:
        self.getNpTransitionMatrix()
        with open(filePath, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load(filePath)->pickle:
        with open(filePath, 'rb') as file:
            loadedPickle = pickle.load(file)
            return loadedPickle
