import os
import pickle

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from MachineSettings import MachineSettings
from ctumrs.OneAlphabetWordsTransitionMatrix import OneAlphabetWordsTransitionMatrix
from ctumrs.TimePosVelsClusteringStrgy import TimePosVelsClusteringStrgy
import yaml
# To make yaml file load faster
from yaml import CLoader

from mMath.calculus.derivative.TimePosRowsDerivativeComputer import TimePosRowsDerivativeComputer

class LiveLocLidarAbn:
    pass

if __name__ == "__main__":
    #the settings
    targetRobotId = "uav2"
    normalScenarioName = "normal"
    testScenarioName = "normal"
    basePath = MachineSettings.MAIN_PATH+"projs/research/data/self-aware-drones/ctumrs/two-drones/"
    pathToNormalScenario = basePath+"{}-scenario/".format(normalScenarioName)
    pathToNormalScenarioYamlFile = pathToNormalScenario+"uav-1-gps-lidar-uva-2-gps-lidar.yaml"
    pathToTestScenrio = basePath+"{}-scenario/".format(testScenarioName)
    pathToTestScenarioYamlFile = pathToTestScenrio + "uav-1-gps-lidar-uva-2-gps-lidar.yaml"
    clustersNum = 75
    velCo = 25
    dataDim = 1440
    pathToTransMtx = "/home/donkarlo/Desktop/new.pkl"

    #for example for uav1 and
    specificRobotAndTopicRowLimit = 100000

    # normal scenrio trans matrix builder
    if not os.path.exists(pathToTransMtx):
        robotTimeLidarRangesVelsObss = []
        with open(pathToNormalScenarioYamlFile, "r") as file:
            topicRows = yaml.load_all(file, Loader=CLoader)
            specificRobotAndTopicRowCounter = 0
            for topicRowCounter, topicRow in enumerate(topicRows):
                robotId,sensorType = topicRow["header"]["frame_id"].split("/")
                if robotId!=targetRobotId or sensorType!="rplidar":
                    continue
                if specificRobotAndTopicRowCounter >= specificRobotAndTopicRowLimit:
                    break

                specificRobotAndTopicRowCounter += 1
                time = float(str(topicRow["header"]["stamp"]["secs"]) + "." + str(topicRow["header"]["stamp"]["nsecs"]))
                npRanges = np.array(topicRow["ranges"]).astype(float)
                #replace infs with 15 in np.range
                npRanges[npRanges == np.inf]=15
                robotTimeLidarRangesVelsObss.append(np.insert(npRanges, 0, time, axis=0))

        robotTimeLidarRangesVelsObss = np.asarray(robotTimeLidarRangesVelsObss)
        robotTimeLidarRangesVelsObss = TimePosRowsDerivativeComputer.computer(robotTimeLidarRangesVelsObss,velCo)

        posVelObssClusteringStrgy = TimePosVelsClusteringStrgy(clustersNum, robotTimeLidarRangesVelsObss[:, 1:dataDim + 1])
        oneAlphabetWordsTransitionMatrix = OneAlphabetWordsTransitionMatrix(posVelObssClusteringStrgy, robotTimeLidarRangesVelsObss[:, 1:dataDim + 1])
        oneAlphabetWordsTransitionMatrix.getNpTransitionMatrix()
        oneAlphabetWordsTransitionMatrix.save(pathToTransMtx)
    else:
        with open(pathToTransMtx, 'rb') as file:
            oneAlphabetWordsTransitionMatrix = pickle.load(file)

    # test scenario abn compute
    mpl.use("TkAgg")
    plt.ion()
    fig, axes = plt.subplots(2, 1)
    line1, = axes[0].plot(np.random.randn(100))
    line2, = axes[1].plot(np.random.randn(100))
    plt.show(block=False)


    def updateGpsPlot(robotSpecificGpsTimeRows):
        line1.set_xdata( robotSpecificGpsTimeRows[:, 1])
        line1.set_ydata( robotSpecificGpsTimeRows[:, 2])
        axes[0].relim()
        axes[0].autoscale_view()
        fig.canvas.draw()
        fig.canvas.flush_events()

    def updateLidarAbnPlot(lidarAbnVals):
        line2.set_xdata(range(0,len(lidarAbnVals)))
        line2.set_ydata(lidarAbnVals)
        axes[1].relim()
        axes[1].autoscale_view()
        fig.canvas.draw()
        fig.canvas.flush_events()


    robotSpecificLidarCounterLimit = 6000
    robotSpecificLidarCounter = 0
    robotSpecificGpsTimeRows = []
    robotSpecificLidarTimeRangesRows = []
    robotSpecificLidarTimeRangesVelsRows = np.empty([1,dataDim+1])
    abnormalityValues = []

    with open(pathToTestScenarioYamlFile, "r") as file:
        topicRows = yaml.load_all(file, Loader=CLoader)
        specificRobotAndTopicRowCounter = 0
        for topicRowCounter, topicRow in enumerate(topicRows):
            robotId, sensorType = topicRow["header"]["frame_id"].split("/")
            if robotId != targetRobotId:
                continue

            if robotSpecificLidarCounter >= robotSpecificLidarCounterLimit:
                break

            time = float(str(topicRow["header"]["stamp"]["secs"]) + "." + str(topicRow["header"]["stamp"]["nsecs"]))

            if sensorType == "gps_origin":
                gpsX = float(topicRow["pose"]["pose"]["position"]["x"])
                gpsY = float(topicRow["pose"]["pose"]["position"]["y"])
                robotSpecificGpsTimeRows.append([time, gpsX, gpsY])
                updateGpsPlot(np.asarray(robotSpecificGpsTimeRows))
            elif sensorType == "rplidar":
                npRanges = np.array(topicRow["ranges"]).astype(float)
                npRanges[npRanges == np.inf] = 15
                robotSpecificLidarTimeRangesRows.append(np.insert(npRanges, 0, time, axis=0))
                if robotSpecificLidarCounter == 0:
                    robotSpecificLidarCounter += 1
                    continue
                if robotSpecificLidarCounter >= 1:
                    prvTime = robotSpecificLidarTimeRangesRows[robotSpecificLidarCounter - 1][0]
                    curTime = robotSpecificLidarTimeRangesRows[robotSpecificLidarCounter][0]
                    diffTime = curTime - prvTime

                    prvPos = robotSpecificLidarTimeRangesRows[robotSpecificLidarCounter - 1][1:]
                    curPos = robotSpecificLidarTimeRangesRows[robotSpecificLidarCounter][1:]
                    diffPos = np.subtract(curPos, prvPos)

                    curVel = velCo * diffPos / diffTime
                    curTimePosVel = np.hstack(np.array([curTime, curPos, curVel], dtype=object))
                    robotSpecificLidarTimeRangesVelsRows =np.vstack([robotSpecificLidarTimeRangesVelsRows,curTimePosVel])
                if robotSpecificLidarCounter == 1:
                    robotSpecificLidarTimeRangesVelsRows = np.delete(robotSpecificLidarTimeRangesVelsRows,0,0)
                    arrToAdd = np.hstack(np.array([prvTime,prvPos,curVel],dtype=object))
                    robotSpecificLidarTimeRangesVelsRows= np.insert(robotSpecificLidarTimeRangesVelsRows,0,arrToAdd,axis=0)

                robotSpecificLidarTimeRangesVelsRows = np.array(robotSpecificLidarTimeRangesVelsRows)
                curObs = robotSpecificLidarTimeRangesVelsRows[-1,1:]
                prvObs = robotSpecificLidarTimeRangesVelsRows[-2,1:]


                #abn computer
                prvObsLabel = oneAlphabetWordsTransitionMatrix.getClusteringStrgy().getPredictedLabelByPosVelObs(prvObs)
                predictedNextLabel = oneAlphabetWordsTransitionMatrix.getHighestPorobabelNextLabelBasedOnthePrvOne(
                    prvObsLabel)
                predictedNextLabelCenter = oneAlphabetWordsTransitionMatrix.getClusteringStrgy().getClusterCenterByLabel(
                    predictedNextLabel)
                abnormalityValue = np.linalg.norm(np.array(curObs) - np.array(predictedNextLabelCenter))
                print(abnormalityValue)
                abnormalityValues.append(abnormalityValue)

                updateLidarAbnPlot(np.array(abnormalityValues))
                robotSpecificLidarCounter += 1




