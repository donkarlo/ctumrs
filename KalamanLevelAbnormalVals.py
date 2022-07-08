import numpy as np

from ctumrs.TimePosVelsClusteringStrgy import TimePosVelsClusteringStrgy

class GaussianNoise:
    def __init__(self):
        pass

class PosVelObsModel:
    def __init__(self,noise:GaussianNoise):
        self.__noise = noise
    def getPosVelObsByState(self,state:np.ndarray)->np.ndarray:
        return self.getObsMtxByState(state).dot(state)

    def getObsMtxByState(self,state:np.ndarray)->np.ndarray:
        rowsNum = state.shape[0]
        obsMtx:np.ndarray = np.zeros((rowsNum,rowsNum))
        for rowCounter in range(1,rowsNum/2):
            obsMtx[rowCounter][rowCounter] = 1
        return obsMtx




class ProcessModel:

    def __init__(self,noise:GaussianNoise):
        self.__noise = noise

    def getPredictedState(self,prvState:np.ndarray,deltaT:float):
        prMtx:np.ndarray = self.getProcessMtxByStateAndDeltaT(prvState, deltaT)
        return prMtx.dot(prvState)


    def getProcessMtxByStateAndDeltaT(sel, sampleState:np.ndarray, deltaT:float):
        rowsNum = sampleState.shape[0]
        processMatrix = np.identity(sampleState.shape[0])
        for rowCounter in range(1, rowsNum / 2):
            processMatrix[rowCounter - 1][rowsNum / 2 + rowCounter] = deltaT
        return processMatrix

    def getNoise(self):
        return self.__noise





class KalamanFilter:
    def __init__(self):
        pass


if __name__ == "__main__":
    # load drone, sensor specific obss
    robotPosVelObss = None

    # cluster data
    posVelClusteringStrgy:TimePosVelsClusteringStrgy = None

    # loop therough obss data
    repeatedLabels = []
    for robotPosVelObsCounter,curRobotPosVelObs in enumerate(robotPosVelObss):
        labelsHist = posVelClusteringStrgy.getLabelByPosVelObs(curRobotPosVelObs)
        labelsHist.append(labelsHist)
        curClusterVelCenter = posVelClusteringStrgy.getClueterVelCenterByLabel(labelsHist)
        if robotPosVelObsCounter == 0:
            pass
        if robotPosVelObsCounter >= 1:
            prvRobotPosVelObs = robotPosVelObss[robotPosVelObsCounter-1]
            prvClusterLabel = posVelClusteringStrgy.getLabelByPosVelObs(prvRobotPosVelObs)
            prvClusterVelCenter = posVelClusteringStrgy.getClueterVelCenterByLabel(prvClusterLabel)


