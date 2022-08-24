import os
import pickle

import numpy as np

from MachineSettings import MachineSettings
from ctumrs.OneAlphabetWordsTransitionMatrix import OneAlphabetWordsTransitionMatrix
from ctumrs.TimePosVelsClusteringStrgy import TimePosVelsClusteringStrgy
import yaml
# To make yaml file load faster
from yaml import CLoader

from ctumrs.sensors.liveLocSensorAbn.one.PlotAll import PlotAll
from ctumrs.topic.GpsOrigin import GpsOrigin
from ctumrs.topic.RpLidar import RpLidar
from ctumrs.topic.Topic import Topic
from mMath.calculus.derivative.TimePosRowsDerivativeComputer import TimePosRowsDerivativeComputer

class LiveLocLidarAbn:
    pass

if __name__ == "__main__":
    #the settings
    targetRobotId = "uav2"
    normalScenarioName = "normal"
    testScenarioName = "follow"
    basePath = MachineSettings.MAIN_PATH+"projs/research/data/self-aware-drones/ctumrs/two-drones/"
    pathToNormalScenario = basePath+"{}-scenario/".format(normalScenarioName)
    pathToNormalScenarioYamlFile = pathToNormalScenario+"uav1-gps-lidar-uav2-gps-lidar.yaml"


    #Lidar settings
    lidarSensorName = "rplidar"
    lidarClustersNum = 75
    lidarVelCo = 25
    lidarRangesVelsDim = 1440
    # for example for uav1 and lidar
    targetRobotLidarTopicRowLimit = 60023
    pathToLidarTransMtx = "/home/donkarlo/Desktop/transition_matrix_scenario_{}_robotid_{}_sensor_{}_training_{}_velco_{}_clusters_{}.pkl".format(
        normalScenarioName
        , targetRobotId
        , lidarSensorName
        , targetRobotLidarTopicRowLimit
        , lidarVelCo
        , lidarClustersNum)

    #Gps settings
    gpsSensorName = "gps_origin"
    gpsVelCo = 25
    gpsVelsDim = 6
    gpsClustersNum = 75
    gpsUpdateRate = 0.01
    targetRobotGpsTopicRowLimit = 300000
    pathToGpsTransMtx = "/home/donkarlo/Desktop/transition_matrix_scenario_{}_robotid_{}_sensor_{}_training_{}_velco_{}_clusters_{}.pkl".format(
        normalScenarioName
        , targetRobotId
        , gpsSensorName
        , targetRobotGpsTopicRowLimit
        , gpsVelCo
        , gpsClustersNum)


    # normal scenrio lidar and gps trans matrix builder if not exist
    if not os.path.exists(pathToLidarTransMtx) or not os.path.exists(pathToGpsTransMtx):

        with open(pathToNormalScenarioYamlFile, "r") as file:
            topicRows = yaml.load_all(file, Loader=CLoader)

            robotTimeLidarRangesVelsObss = []
            robotTimeGpsVelsObss = []
            targetRobotLidarTopicRowCounter = 0
            targetRobotGpsTopicRowCounter = 0

            for topicRowCounter, topicRow in enumerate(topicRows):
                if targetRobotLidarTopicRowCounter >= targetRobotLidarTopicRowLimit and targetRobotGpsTopicRowCounter>=targetRobotGpsTopicRowLimit:
                    break
                robotId, sensorName = Topic.getRobotIdAndSensorName(topicRow)
                if robotId!=targetRobotId:
                    continue

                time = Topic.staticGetTimeByTopicDict(topicRow)

                if targetRobotLidarTopicRowCounter < targetRobotLidarTopicRowLimit:
                    if not os.path.exists(pathToLidarTransMtx):
                        if sensorName==lidarSensorName:
                            npRanges = RpLidar.staticGetNpRanges(topicRow)
                            robotTimeLidarRangesVelsObss.append(np.insert(npRanges, 0, time, axis=0))
                            targetRobotLidarTopicRowCounter += 1
                            print("robot: {}, Sensor: {}, count: {}".format(robotId, sensorName
                                                                            ,targetRobotLidarTopicRowCounter))
                if targetRobotGpsTopicRowCounter < targetRobotGpsTopicRowLimit:
                    if not os.path.exists(pathToGpsTransMtx):
                        if sensorName==gpsSensorName:
                            gpsX,gpsY,gpsZ = GpsOrigin.staticGetXyz(topicRow)
                            robotTimeGpsVelsObss.append([time, gpsX, gpsY, gpsZ])
                            targetRobotGpsTopicRowCounter += 1
                            print("robot: {}, Sensor: {}, count: {}".format(robotId, sensorName,
                                                                            targetRobotGpsTopicRowCounter))

        # calculating transMtx for rplidar
        if not os.path.exists(pathToLidarTransMtx):
            robotTimeLidarRangesVelsObss = np.asarray(robotTimeLidarRangesVelsObss)
            robotTimeLidarRangesVelsObss = TimePosRowsDerivativeComputer.computer(robotTimeLidarRangesVelsObss, lidarVelCo)

            #clustering
            lidarRangesVelsObssClusteringStrgy = TimePosVelsClusteringStrgy(lidarClustersNum
                                                                            , robotTimeLidarRangesVelsObss[:, 1:lidarRangesVelsDim + 1])
            lidarRangesVelsObssClusteringStrgyDict = lidarRangesVelsObssClusteringStrgy.getLabeledTimePosVelsClustersDict(robotTimeLidarRangesVelsObss[:, 1:lidarRangesVelsDim + 1])
            #Building Lidar transition matrix
            lidarOneAlphabetWordsTransitionMatrix = OneAlphabetWordsTransitionMatrix(lidarRangesVelsObssClusteringStrgy
                                                                                     , robotTimeLidarRangesVelsObss[:, 1:lidarRangesVelsDim + 1])
            lidarOneAlphabetWordsTransitionMatrix.getNpTransitionMatrix()
            lidarOneAlphabetWordsTransitionMatrix.save(pathToLidarTransMtx)
        else:
            with open(pathToLidarTransMtx, 'rb') as file:
                lidarOneAlphabetWordsTransitionMatrix = pickle.load(file)

        # calculating transMtx for GPS
        if not os.path.exists(pathToGpsTransMtx):
            robotTimeGpsVelsObss = np.asarray(robotTimeGpsVelsObss)
            robotTimeGpsVelsObss = TimePosRowsDerivativeComputer.computer(robotTimeGpsVelsObss,
                                                                                  gpsVelCo,gpsUpdateRate)

            # clustering gps data
            gpsVelsObssClusteringStrgy = TimePosVelsClusteringStrgy(gpsClustersNum
                                                                    , robotTimeGpsVelsObss[:,
                                                                              1:gpsVelsDim + 1])
            gpsVelsObssClusteringStrgyDict = gpsVelsObssClusteringStrgy.getLabeledTimePosVelsClustersDict(
                robotTimeGpsVelsObss[:, 1:gpsVelsDim + 1])
            # Building Gps transition matrix
            gpsOneAlphabetWordsTransitionMatrix = OneAlphabetWordsTransitionMatrix(gpsVelsObssClusteringStrgy
                                                                                   , robotTimeGpsVelsObss[:,
                                                                                       1:gpsVelsDim + 1])
            gpsOneAlphabetWordsTransitionMatrix.getNpTransitionMatrix()
            gpsOneAlphabetWordsTransitionMatrix.save(pathToGpsTransMtx)
        else:
            with open(pathToGpsTransMtx, 'rb') as file:
                gpsOneAlphabetWordsTransitionMatrix = pickle.load(file)
    else:
        with open(pathToLidarTransMtx, 'rb') as file:
            lidarOneAlphabetWordsTransitionMatrix = pickle.load(file)
        with open(pathToGpsTransMtx, 'rb') as file:
            gpsOneAlphabetWordsTransitionMatrix = pickle.load(file)



    # From here we gather data for gps location and  lidar abnormalities and gps abnormalities
    pathToTestScenrio = basePath + "{}-scenario/".format(testScenarioName)
    pathToTestScenarioYamlFile = pathToTestScenrio + "uav1-gps-lidar-uav2-gps-lidar.yaml"

    #Lidar settings
    targetRobotLidarCounterLimit = 6000
    targetRobotLidarCounter = 0
    targetRobotLidarTimeRangesObss = []
    targetRobotLidarTimeRangesVelsObss = np.empty([1, lidarRangesVelsDim + 1])
    lidarTimeAbnormalityValues = []

    #GPS settings
    targetRobotGpsTimeRows = []
    targetRobotGpsCounter = 0
    targetRobotGpsTimeVelsObss = []
    targetRobotGpsTimeRangesVelsObss = np.empty([1, gpsVelsDim + 1])
    gpsTimeAbnormalityValues = []



    plotAll = PlotAll()



    with open(pathToTestScenarioYamlFile, "r") as file:
        topicRows = yaml.load_all(file, Loader=CLoader)
        targetRobotLidarTopicRowCounter = 0
        for topicRowCounter, topicRow in enumerate(topicRows):
            if targetRobotLidarCounter >= targetRobotLidarCounterLimit:
                break

            robotId, sensorName = Topic.getRobotIdAndSensorName(topicRow)
            if robotId != targetRobotId:
                continue

            time = Topic.staticGetTimeByTopicDict(topicRow)

            if sensorName == "gps_origin":
                gpsX,gpsY,gpsZ = GpsOrigin.staticGetXyz(topicRow)

                if targetRobotGpsCounter == 0:
                    targetRobotGpsTimeXyzVelsObss = np.asarray([[time, gpsX, gpsY, gpsZ,0,0,0]])
                    targetRobotGpsCounter += 1
                    continue
                if targetRobotGpsCounter >= 1:
                    targetRobotGpsTimeXyzVelsObss = np.vstack((targetRobotGpsTimeXyzVelsObss, [time, gpsX, gpsY, gpsZ,0,0,0]))
                    timeDiff = targetRobotGpsTimeXyzVelsObss[-1][0]-targetRobotGpsTimeXyzVelsObss[-2][0]
                    if timeDiff == 0:
                        timeDiff = gpsUpdateRate
                    diffGps = np.subtract(targetRobotGpsTimeXyzVelsObss[-1][1:int(gpsVelsDim/2)],targetRobotGpsTimeXyzVelsObss[-2][1:int(gpsVelsDim/2)])
                    gpsVels = gpsVelCo*diffGps/timeDiff
                    targetRobotGpsTimeXyzVelsObss[-1][int(gpsVelsDim/2)+1:gpsVelsDim]=gpsVels

                plotAll.updateGpsPlot(np.asarray(targetRobotGpsTimeXyzVelsObss))
                targetRobotGpsCounter += 1

                # gps abn computer
                gpsCurObs = targetRobotGpsTimeXyzVelsObss[-1, 1:]
                gpsPrvObs = targetRobotGpsTimeXyzVelsObss[-2, 1:]
                gpsPrvObsLabel = gpsOneAlphabetWordsTransitionMatrix.getClusteringStrgy().getPredictedLabelByPosVelObs(
                    gpsPrvObs)
                gpsPredictedNextLabel = gpsOneAlphabetWordsTransitionMatrix.getHighestPorobabelNextLabelBasedOnthePrvOne(
                    gpsPrvObsLabel)
                gpsPredictedNextLabelCenter = gpsOneAlphabetWordsTransitionMatrix.getClusteringStrgy().getClusterCenterByLabel(
                    gpsPredictedNextLabel)
                gpsAbnormalityValue = np.linalg.norm(
                    np.array(gpsCurObs) - np.array(gpsPredictedNextLabelCenter))
                print("Gps abnormality value: " + str(gpsAbnormalityValue))
                gpsTimeAbnormalityValues.append([time, gpsAbnormalityValue])

                plotAll.updateGpsAbnPlot(np.array(gpsTimeAbnormalityValues))


            elif sensorName == "rplidar":
                npRanges = RpLidar.staticGetNpRanges(topicRow)
                targetRobotLidarTimeRangesObss.append(np.insert(npRanges, 0, time, axis=0))
                if targetRobotLidarCounter == 0:
                    targetRobotLidarCounter += 1
                    continue
                if targetRobotLidarCounter >= 1:
                    prvLidarTime = targetRobotLidarTimeRangesObss[targetRobotLidarCounter - 1][0]
                    curLidarTime = targetRobotLidarTimeRangesObss[targetRobotLidarCounter][0]
                    diffLidarTime = curLidarTime - prvLidarTime

                    prvLidarRanges = targetRobotLidarTimeRangesObss[targetRobotLidarCounter - 1][1:]
                    curLidarRanges = targetRobotLidarTimeRangesObss[targetRobotLidarCounter][1:]
                    diffLidarRanges = np.subtract(curLidarRanges, prvLidarRanges)

                    curLidarVel = lidarVelCo * diffLidarRanges / diffLidarTime
                    curLidarTimeRangesVels = np.hstack(np.array([curLidarTime, curLidarRanges, curLidarVel], dtype=object))
                    targetRobotLidarTimeRangesVelsObss =np.vstack([targetRobotLidarTimeRangesVelsObss, curLidarTimeRangesVels])
                if targetRobotLidarCounter == 1:
                    targetRobotLidarTimeRangesVelsObss = np.delete(targetRobotLidarTimeRangesVelsObss, 0, 0)
                    liarRangesVelsToAdd = np.hstack(np.array([prvLidarTime, prvLidarRanges, curLidarVel], dtype=object))
                    targetRobotLidarTimeRangesVelsObss= np.insert(targetRobotLidarTimeRangesVelsObss, 0, liarRangesVelsToAdd, axis=0)

                targetRobotLidarTimeRangesVelsObss = np.array(targetRobotLidarTimeRangesVelsObss)

                # lidar abn computer
                lidarCurObs = targetRobotLidarTimeRangesVelsObss[-1, 1:]
                lidarPrvObs = targetRobotLidarTimeRangesVelsObss[-2, 1:]
                lidarPrvObsLabel = lidarOneAlphabetWordsTransitionMatrix.getClusteringStrgy().getPredictedLabelByPosVelObs(lidarPrvObs)
                lidarPredictedNextLabel = lidarOneAlphabetWordsTransitionMatrix.getHighestPorobabelNextLabelBasedOnthePrvOne(
                    lidarPrvObsLabel)
                lidarPredictedNextLabelCenter = lidarOneAlphabetWordsTransitionMatrix.getClusteringStrgy().getClusterCenterByLabel(
                    lidarPredictedNextLabel)
                lidarAbnormalityValue = np.linalg.norm(np.array(lidarCurObs) - np.array(lidarPredictedNextLabelCenter))
                print("Lidar abnormality value: " + str(lidarAbnormalityValue))
                lidarTimeAbnormalityValues.append([time, lidarAbnormalityValue])

                plotAll.updateLidarAbnPlot(np.array(lidarTimeAbnormalityValues))
                targetRobotLidarCounter += 1




