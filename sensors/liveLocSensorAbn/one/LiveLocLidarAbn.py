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
from ctumrs.topic.Time import Time
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
    targetRobotLidarTopicRowLimit = 6000
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
    targetRobotGpsTopicRowLimit = targetRobotLidarTopicRowLimit * 3
    pathToGpsTransMtx = "/home/donkarlo/Desktop/transition_matrix_scenario_{}_robotid_{}_sensor_{}_training_{}_velco_{}_clusters_{}.pkl".format(
        normalScenarioName
        , targetRobotId
        , gpsSensorName
        , targetRobotGpsTopicRowLimit
        , gpsVelCo
        , gpsClustersNum)


    # normal scenrio trans matrix builder
    if not os.path.exists(pathToLidarTransMtx) or not os.path.exists(pathToGpsTransMtx):
        robotTimeLidarRangesVelsObss = []
        robotTimeGpsVelsObss = []
        with open(pathToNormalScenarioYamlFile, "r") as file:
            topicRows = yaml.load_all(file, Loader=CLoader)
            targetRobotLidarTopicRowCounter = 0
            targetRobotGpsTopicRowCounter = 0
            for topicRowCounter, topicRow in enumerate(topicRows):
                robotId, sensorName = topicRow["header"]["frame_id"].split("/")
                if robotId!=targetRobotId:
                    continue
                if targetRobotLidarTopicRowCounter >= targetRobotLidarTopicRowLimit or targetRobotGpsTopicRowCounter>=targetRobotGpsTopicRowLimit:
                    break

                if sensorName == lidarSensorName:
                    targetRobotLidarTopicRowCounter += 1
                    print("robot: {}, Sensor: {}, count: {}".format(robotId, sensorName, targetRobotLidarTopicRowCounter))

                if sensorName == gpsSensorName:
                    targetRobotGpsTopicRowCounter += 1
                    print("robot: {}, Sensor: {}, count: {}".format(robotId, sensorName, targetRobotGpsTopicRowCounter))

                time = Time.staticFloatTimeFromTopicDict(topicRow)

                if not os.path.exists(pathToLidarTransMtx):
                    if sensorName==lidarSensorName:
                        npRanges = np.array(topicRow["ranges"]).astype(float)
                        #replace infs with 15 in np.range
                        npRanges[npRanges == np.inf]=15
                        robotTimeLidarRangesVelsObss.append(np.insert(npRanges, 0, time, axis=0))
                if not os.path.exists(pathToGpsTransMtx):
                    if sensorName==gpsSensorName:
                        gpsX = float(topicRow["pose"]["pose"]["position"]["x"])
                        gpsY = float(topicRow["pose"]["pose"]["position"]["y"])
                        gpsZ = float(topicRow["pose"]["pose"]["position"]["z"])
                        robotTimeGpsVelsObss.append([time, gpsX, gpsY, gpsZ])

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

            # clustering
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



    plotAll = PlotAll()



    with open(pathToTestScenarioYamlFile, "r") as file:
        topicRows = yaml.load_all(file, Loader=CLoader)
        targetRobotLidarTopicRowCounter = 0
        for topicRowCounter, topicRow in enumerate(topicRows):
            robotId, sensorName = topicRow["header"]["frame_id"].split("/")
            if robotId != targetRobotId:
                continue

            if targetRobotLidarCounter >= targetRobotLidarCounterLimit:
                break


            time = Time.staticFloatTimeFromTopicDict(topicRow)

            if sensorName == "gps_origin":
                gpsX = float(topicRow["pose"]["pose"]["position"]["x"])
                gpsY = float(topicRow["pose"]["pose"]["position"]["y"])
                gpsZ = float(topicRow["pose"]["pose"]["position"]["z"])
                targetRobotGpsTimeRows.append([time, gpsX, gpsY, gpsZ])



                plotAll.updateGpsPlot(np.asarray(targetRobotGpsTimeRows))
                targetRobotGpsCounter += 1
            elif sensorName == "rplidar":
                npRanges = np.array(topicRow["ranges"]).astype(float)
                npRanges[npRanges == np.inf] = 15
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
                lidarCurObs = targetRobotLidarTimeRangesVelsObss[-1, 1:]
                lidarPrvObs = targetRobotLidarTimeRangesVelsObss[-2, 1:]


                #abn computer
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




