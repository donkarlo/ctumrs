import numpy as np
from ctumrs.PosVelsClusteringStrgy import PosVelsClusteringStrgy
import pickle

from ctumrs.TimePosVelObssUtility import TimePosVelObssUtility


class TwoAlphabetWordsTransitionMatrix:
    def __init__(self
                 , robot1TimePosVelClusteringStrgy:PosVelsClusteringStrgy = None
                 , robot2TimePosVelClusteringStrgy:PosVelsClusteringStrgy = None
                 , robot1TimePosVels:list = None
                 , robot2TimePosVels:list = None
                 ):

        self.__robot1PosVelClusteringStrgy: PosVelsClusteringStrgy = robot1TimePosVelClusteringStrgy
        self.__robot2PosVelClusteringStrgy: PosVelsClusteringStrgy = robot2TimePosVelClusteringStrgy

        self.__robot1TimePosVels = robot1TimePosVels
        self.__robot2TimePosVels = robot2TimePosVels

        self.__npTransitionMatrix = None
        self.__mappingRowOrColNumLabelMapDict = None

        self.__robot1And2ObsMatchStrgy = "FIND_BY_TIMESTAMP"

    def setRobot1And2ObsMatchStrgy(self, leaderFollowerObsMatchStrgy="ALREADY_INDEX_MATCHED"):
        self.__robot1And2ObsMatchStrgy = leaderFollowerObsMatchStrgy

    def getRobot1TimePosVelClusteringStrgy(self)->PosVelsClusteringStrgy:
        return self.__robot1PosVelClusteringStrgy

    def getRobot2TimePosVelClusteringStrgy(self)->PosVelsClusteringStrgy:
        return self.__robot2PosVelClusteringStrgy

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
            for i in range(0, self.__robot1PosVelClusteringStrgy.getClustersNum()):
                for j in range(0, self.__robot2PosVelClusteringStrgy.getClustersNum()):
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
            self.__npTransitionMatrix = np.zeros((self.__robot1PosVelClusteringStrgy.getClustersNum() * self.__robot2PosVelClusteringStrgy.getClustersNum()
             , self.__robot1PosVelClusteringStrgy.getClustersNum() * self.__robot2PosVelClusteringStrgy.getClustersNum()))
            foundFollowerLabels = []
            foundLeaderLabels = []
            for leaderUavCurTimePosVelCounter, curLeaderUavTimePosVel in enumerate(self.__robot1TimePosVels):
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
                    self.__robot1PosVelClusteringStrgy.getFittedClusters().predict([TimePosVelObssUtility.getPosVelByTimePosVel(prvLeaderUavTimePosVel)])[0]
                    prvFollowerUavLabel = self.__robot2PosVelClusteringStrgy.getFittedClusters().predict(
                        [TimePosVelObssUtility.getPosVelByTimePosVel(prvClosestFollowerTimePosVel)])[0]

                    # Finding the label for current leader follower  observation
                    curLeaderUavLabel = \
                    self.__robot1PosVelClusteringStrgy.getFittedClusters().predict([TimePosVelObssUtility.getPosVelByTimePosVel(curLeaderUavTimePosVel)])[0]
                    curFollowerUavLabel = self.__robot2PosVelClusteringStrgy.getFittedClusters().predict(
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
