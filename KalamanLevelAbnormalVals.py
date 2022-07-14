import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from ctumrs.TimePosVelObssUtility import TimePosVelObssUtility
from ctumrs.TimePosVelsClusteringStrgy import TimePosVelsClusteringStrgy

class PosObsModel():
    def __init__(self):
        pass

    def getVelRemovedObsMtxByState(self,state:np.ndarray)->np.ndarray:
        rowsNum = state.shape[1]
        obsMtx:np.ndarray = np.zeros((rowsNum,rowsNum))
        for rowCounter in range(1,rowsNum/2):
            obsMtx[rowCounter][rowCounter] = 1
        return obsMtx

class PosVelObsModel:
    def __init__(self,stateSpaceDim:int,noiseCovMtx:np.ndarray):
        self.__stateSpaceDim = stateSpaceDim
        self.__noiseCovMtx = noiseCovMtx

    def getMtx(self)->np.ndarray:
        return np.eye(self.__stateSpaceDim)

    def getNoiseCovMtx(self):
        return self.__noiseCovMtx






class ProcessModel:

    def __init__(self,stateSpaceDim:int,noiseCovMtx:np.ndarray):
        self.__stateSpaceDim = stateSpaceDim
        self.__noiseCovMtx:np.ndarray = noiseCovMtx

    def getPredictedState(self, prvCorrectedState:np.ndarray, deltaT:float):
        prMtx:np.ndarray = self.getProcessMtxByDeltaT(deltaT)
        return prMtx @ prvCorrectedState

    def getProcessMtxByDeltaT(self, deltaT:float):
        rowsNum = self.__stateSpaceDim
        processMatrix = np.identity(rowsNum)
        for rowCounter in range(1, int(rowsNum / 2)):
            processMatrix[rowCounter - 1][int(rowsNum / 2) + rowCounter] = deltaT
        return processMatrix

    def getNoiseCovMtx(self):
        return self.__noiseCovMtx

if __name__ == "__main__":
    # load drone, sensor specific data
    stateSpaceDim = 6

    timePosVelObssUtility = TimePosVelObssUtility()
    jointPathToLeaderAndFollowerAbnormalScenario = "/home/donkarlo/Dropbox/projs/research/data/self-aware-drones/ctumrs/two-drones/follow-scenario/gps/"
    pathToLeaderUavTimePosVelDataFile = jointPathToLeaderAndFollowerAbnormalScenario + "uav-1-cleaned-gps-pos-vel.txt"

    leaderUavPosVelsAndTimePosVels = timePosVelObssUtility.getTimePosVelsAndPosVels(pathToLeaderUavTimePosVelDataFile,
                                                                                    10000)

    robotTimePosVelObss = leaderUavPosVelsAndTimePosVels['timePosVels']


    #add vels

    # cluster data
    posVelClusteringStrgy:TimePosVelsClusteringStrgy = TimePosVelsClusteringStrgy(75,robotTimePosVelObss[:,1:])

    # loop therough obss data
    clusterLabelCounter = 0

    covValPXy = 2*10e-4
    covValPZz = 4*10e-4
    covValVXy = 0.2
    covValVZz = 0.4


    prvCorrectedUncertainty = np.array(
         [
         [covValPXy,0,0,0,0,0]
        ,[0,covValPXy,0,0,0,0]
        ,[0,0,covValPZz,0,0,0]
        ,[0,0,0,covValVXy,0,0]
        ,[0,0,0,0,covValVXy,0]
        ,[0,0,0,0,0,covValVZz]
         ]
    )

    processNoiseCovMtx = prvCorrectedUncertainty
    obsNoiseCovMtx = np.array([
         [covValPXy]
        ,[covValPXy]
        ,[covValPZz]
        ,[covValVXy]
        ,[covValVXy]
        ,[covValVZz]])

    def kalmanFilter():
        innovations = []
        for robotPosVelObsCounter, curRobotTimePosVelObs in enumerate(robotTimePosVelObss):
            if robotPosVelObsCounter<15000:
                if robotPosVelObsCounter == 0:
                    prvTime = curRobotTimePosVelObs[0]
                    prvCorrectedState = curRobotTimePosVelObs[1:].T
                    prvClusterLabel = posVelClusteringStrgy.getPredictedLabelByPosVelObs(curRobotTimePosVelObs[1:])
                    continue
                curTime = curRobotTimePosVelObs[0]
                curPosVelObs= curRobotTimePosVelObs[1:].T

                curClusterLabel = posVelClusteringStrgy.getPredictedLabelByPosVelObs(curRobotTimePosVelObs[1:])
                curClusterVelCenter = posVelClusteringStrgy.getClueterVelCenterByLabel(curClusterLabel)


                if prvClusterLabel!=curClusterLabel:
                    prvCorrectedState = curPosVelObs


                processModel = ProcessModel(stateSpaceDim, processNoiseCovMtx)
                posVelObsModel  = PosVelObsModel(stateSpaceDim,obsNoiseCovMtx)

                ##############prediction
                curPredictedState = processModel.getPredictedState(prvCorrectedState,curTime-prvTime)
                curPredictedUncertainty = processModel.getProcessMtxByDeltaT(curTime-prvTime) @ prvCorrectedUncertainty @ processModel.getProcessMtxByDeltaT(curTime-prvTime).T + processModel.getNoiseCovMtx()

                #############update
                curGain = curPredictedUncertainty @ posVelObsModel.getMtx().T @ np.linalg.inv((posVelObsModel.getMtx() @ curPredictedUncertainty @ posVelObsModel.getMtx().T + posVelObsModel.getNoiseCovMtx()))
                innovation = curPosVelObs - posVelObsModel.getMtx() @ curPredictedState
                curCorrectedState = curPredictedState + curGain @ innovation
                curCorrectedUncertainty = (1 - curGain @ posVelObsModel.getMtx()) @ curPredictedState

                print(robotPosVelObsCounter, " ", np.linalg.norm(innovation))
                innovations.append(np.linalg.norm(innovation))

                #store the data that you need
                prvCorrectedState = curCorrectedState
                prvTime= curTime
                prvClusterLabel = curClusterLabel


            else:
                break

        return innovations


    innovations = kalmanFilter()
    plt.plot(innovations)
    plt.show()




