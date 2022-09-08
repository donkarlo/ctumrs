import os
import pickle
import numpy as np
from MachineSettings import MachineSettings
from ctumrs.OneAlphabetWordsTransitionMatrix import OneAlphabetWordsTransitionMatrix
from ctumrs.PosVelsClusteringStrgy import PosVelsClusteringStrgy
import yaml
from yaml import CLoader

from ctumrs.TwoAlphabetWordsClusterLevelAbnormalVals import TwoAlphabetWordsClusterLevelAbnormalVals
from ctumrs.TwoAlphabetWordsTransitionMatrix import TwoAlphabetWordsTransitionMatrix
from ctumrs.sensors.lidar.Autoencoder import Autoencoder
from ctumrs.sensors.liveLocSensorAbn.two.PlotAll import PlotAll
from ctumrs.sensors.liveLocSensorAbn.two.TopicLoopingLogic import TopicLoopingLogic
from ctumrs.topic.GpsOrigin import GpsOrigin
from ctumrs.topic.RpLidar import RpLidar
from ctumrs.topic.Topic import Topic
from mMath.calculus.derivative.TimePosRowsDerivativeComputer import TimePosRowsDerivativeComputer
from mMath.statistic.Distance import Distance

if __name__ == "__main__":
    #configs
    with open("configs.yaml", "r") as file:
        configs = yaml.load(file, Loader=CLoader)

    targetRobotIds = configs["targetRobotIds"]
    normalScenarioName = configs["normalScenarioName"]
    basePath = MachineSettings.MAIN_PATH+"projs/research/data/self-aware-drones/ctumrs/two-drones/"
    pathToNormalScenario = basePath+"{}-scenario/".format(normalScenarioName)
    pathToNormalScenarioYamlFile = pathToNormalScenario+"uav1-gps-lidar-uav2-gps-lidar.yaml"


    #Lidar settings
    lidarSensorName = "rplidar"
    lidarClustersNum = configs["rplidar"]["clustersNum"]
    lidarVelCo = configs["rplidar"]["velCo"]
    lidarTrainingRowsNumLimit = 2*configs["rplidar"]["trainingRowsNumLimit"]
    lidarAutoencoderLatentDim = configs["rplidar"]["autoencoder"]["latentDim"]
    lidarAutoencoderEpochs = configs["rplidar"]["autoencoder"]["epocs"]
    lidarAutoencoderBatchSize = configs["rplidar"]["autoencoder"]["batchSize"]

    lidarRangesDim = 720
    lidarRangesVelsDim = lidarRangesDim*2
    pathToLidarTwoAlphaTransMtxDir = pathToNormalScenario + lidarSensorName + "/"

    #this path will be used to keep autoencodder.h5, encoder.h5 and decoder.h5 for uav1(the leader) if does not exist
    pathToRobot1LidarMindDir = pathToNormalScenario + "{}/{}_mind_training_{}_velco_{}_clusters_{}_autoencoder_latentdim_{}_epochs_{}/".format(
        targetRobotIds[0]#uav1
        , lidarSensorName
        , lidarTrainingRowsNumLimit
        , lidarVelCo
        , lidarClustersNum
        , lidarAutoencoderLatentDim
        , lidarAutoencoderEpochs
    )

    # this path will be used to keep autoencodder.h5, encoder.h5 and decoder.h5 for uav2 if does not exist
    pathToRobot2LidarMindDir = pathToNormalScenario + "{}/{}_mind_training_{}_velco_{}_clusters_{}_autoencoder_latentdim_{}_epochs_{}/".format(
        targetRobotIds[1]#uav2
        , lidarSensorName
        , lidarTrainingRowsNumLimit
        , lidarVelCo
        , lidarClustersNum
        , lidarAutoencoderLatentDim
        , lidarAutoencoderEpochs
    )

    pathToLidarTwoAlphTransMtxFile = pathToLidarTwoAlphaTransMtxDir + "transMtx_training_{}_velco_{}_clusters_{}_autoencoder_latentdim_{}_epochs_{}.pkl".format(
        lidarTrainingRowsNumLimit
        , lidarVelCo
        , lidarClustersNum
        , lidarAutoencoderLatentDim
        , lidarAutoencoderEpochs
    )

    #Gps settings
    gpsSensorName = "gps_origin"
    gpsVelCo = configs["gps_origin"]["velCo"]
    gpsClustersNum = configs["gps_origin"]["clustersNum"]
    gpsTrainingRowsNumLimit = 2*configs["gps_origin"]["trainingRowsNumLimit"]
    gpsVelsDim = 6
    pathToGpsTwoAlphaTransMtxDir = pathToNormalScenario + gpsSensorName + "/"
    pathToGpsTwoAlphTransMtxFile = pathToGpsTwoAlphaTransMtxDir + "transMtx_training_{}_velco_{}_clusters_{}.pkl".format(
        gpsTrainingRowsNumLimit
         , gpsVelCo
         , gpsClustersNum
         )

    # normal scenrio lidar and gps trans matrix builder if not exist
    if TopicLoopingLogic.getShouldLoopThroughTopics(pathToGpsTwoAlphTransMtxFile, pathToLidarTwoAlphTransMtxFile):
        with open(pathToNormalScenarioYamlFile, "r") as file:
            topicRows = yaml.load_all(file, Loader=CLoader)

            robot1TimeLowDimLidarRangesVelsObss = []
            robot2TimeLowDimLidarRangesVelsObss = []
            robot1TimeGpsVelsObss = []
            robot2TimeGpsVelsObss = []
            targetRobotLidarTopicRowCounter = 0
            targetRobotsGpsTopicRowCounter = 0

            for topicRowCounter, topicRow in enumerate(topicRows):
                if (targetRobotLidarTopicRowCounter >= lidarTrainingRowsNumLimit or os.path.exists(pathToLidarTwoAlphTransMtxFile)) \
                        and (targetRobotsGpsTopicRowCounter >= gpsTrainingRowsNumLimit or os.path.exists(pathToGpsTwoAlphTransMtxFile)):
                    break
                robotId, sensorName = Topic.staticGetRobotIdAndSensorName(topicRow)

                time = Topic.staticGetTimeByTopicDict(topicRow)

                if targetRobotsGpsTopicRowCounter < gpsTrainingRowsNumLimit:
                    if not os.path.exists(pathToGpsTwoAlphTransMtxFile):
                        if sensorName==gpsSensorName:
                            gpsX,gpsY,gpsZ = GpsOrigin.staticGetXyz(topicRow)

                            if robotId == targetRobotIds[0]:
                                robot1TimeGpsVelsObss.append([time, gpsX, gpsY, gpsZ])
                            elif robotId == targetRobotIds[1]:
                                robot2TimeGpsVelsObss.append([time, gpsX, gpsY, gpsZ])

                            targetRobotsGpsTopicRowCounter += 1
                            print("robot: {}, Sensor: {}, count: {}".format(robotId
                                                                            , sensorName,
                                                                            targetRobotsGpsTopicRowCounter))

                if targetRobotLidarTopicRowCounter < lidarTrainingRowsNumLimit:
                    if not os.path.exists(pathToLidarTwoAlphTransMtxFile):
                        if sensorName==lidarSensorName:
                            npRanges = RpLidar.staticGetNpRanges(topicRow)

                            if robotId == targetRobotIds[0]:
                                robot1TimeLowDimLidarRangesVelsObss.append(np.insert(npRanges, 0, time, axis=0))
                            elif robotId == targetRobotIds[1]:
                                robot2TimeLowDimLidarRangesVelsObss.append(np.insert(npRanges, 0, time, axis=0))

                            targetRobotLidarTopicRowCounter += 1
                            print("robot: {}, Sensor: {}, count: {}".format(robotId
                                                                            ,sensorName
                                                                         ,targetRobotLidarTopicRowCounter))

        ########### calculating transMtx for GPS
        if not os.path.exists(pathToGpsTwoAlphTransMtxFile):

            robot1TimeGpsVelsObss = np.asarray(robot1TimeGpsVelsObss)
            robot1TimeGpsVelsObss = TimePosRowsDerivativeComputer.computer(robot1TimeGpsVelsObss,gpsVelCo)
            robot1GpsVelsObssClusteringStrgy = PosVelsClusteringStrgy(gpsClustersNum
                                                                      , robot1TimeGpsVelsObss[:, 1:gpsVelsDim + 1])

            robot2TimeGpsVelsObss = np.asarray(robot2TimeGpsVelsObss)
            robot2TimeGpsVelsObss = TimePosRowsDerivativeComputer.computer(robot2TimeGpsVelsObss,gpsVelCo)
            robot2GpsVelsObssClusteringStrgy = PosVelsClusteringStrgy(gpsClustersNum
                                                                      , robot2TimeGpsVelsObss[:,1:gpsVelsDim + 1])

            gpsTwoAlphWordsTransMtx = TwoAlphabetWordsTransitionMatrix(robot1GpsVelsObssClusteringStrgy
                                                                         , robot2GpsVelsObssClusteringStrgy
                                                                         , robot1TimeGpsVelsObss
                                                                         , robot2TimeGpsVelsObss
                                                                         )
            gpsTwoAlphWordsTransMtx.save(pathToGpsTwoAlphTransMtxFile)


        else:
           gpsTwoAlphWordsTransMtx = TwoAlphabetWordsTransitionMatrix.load(pathToGpsTwoAlphTransMtxFile)

        #############LIDAR trans mtx building
        #encoding and saving robot 1 lidar data
        if not os.path.exists(pathToRobot1LidarMindDir):
            os.mkdir(pathToRobot1LidarMindDir)
            robot1TimeLowDimLidarRangesVelsObss = np.asarray(robot1TimeLowDimLidarRangesVelsObss)
            #train the auto encoder for robot 1
            autoencoder = Autoencoder(robot1TimeLowDimLidarRangesVelsObss[:, 1:]
                                      , lidarAutoencoderLatentDim
                                      , lidarAutoencoderEpochs)
            autoencoder.saveFittedAutoencoder(pathToRobot1LidarMindDir + "encoder-decoder.h5")
            autoencoder.saveFittedEncoder(pathToRobot1LidarMindDir + "encoder.h5")
            autoencoder.saveFittedDecoder(pathToRobot1LidarMindDir + "decoder.h5")
            robot1LidarLowDimObss = autoencoder.getPredictedLowDimObss(robot1TimeLowDimLidarRangesVelsObss[:, 1:])
            robot1LidarLowDimTimeObss = np.hstack((robot1TimeLowDimLidarRangesVelsObss[0:, 0:1]
                                                   , robot1LidarLowDimObss))
            #compute velocities
            robot1TimeLowDimLidarRangesVelsObss = np.asarray(TimePosRowsDerivativeComputer.computer(robot1LidarLowDimTimeObss
                                                                                         , lidarVelCo))

        #encoding and saving robot 2 lidar data
        if not os.path.exists(pathToRobot2LidarMindDir):
            os.mkdir(pathToRobot2LidarMindDir)
            robot2TimeLowDimLidarRangesVelsObss = np.asarray(robot2TimeLowDimLidarRangesVelsObss)
            # train the auto encoder for robot 2
            autoencoder = Autoencoder(robot2TimeLowDimLidarRangesVelsObss[:, 1:]
                                      , lidarAutoencoderLatentDim
                                      , lidarAutoencoderEpochs)
            autoencoder.saveFittedAutoencoder(pathToRobot2LidarMindDir + "encoder-decoder.h5")
            autoencoder.saveFittedEncoder(pathToRobot2LidarMindDir + "encoder.h5")
            autoencoder.saveFittedDecoder(pathToRobot2LidarMindDir + "decoder.h5")
            robot2LidarLowDimObss = autoencoder.getPredictedLowDimObss(robot2TimeLowDimLidarRangesVelsObss[:, 1:])
            robot2LidarLowDimTimeObss = np.hstack((robot2TimeLowDimLidarRangesVelsObss[0:, 0:1]
                                                   , robot2LidarLowDimObss))
            # compute velocities
            robot2TimeLowDimLidarRangesVelsObss = np.asarray(TimePosRowsDerivativeComputer.computer(robot2LidarLowDimTimeObss,
                                                                                         lidarVelCo))
        if not os.path.exists(pathToLidarTwoAlphTransMtxFile):
            #cluster each
            robot1LidarClusteringStrgy =  PosVelsClusteringStrgy(lidarClustersNum
                                                                 , robot1TimeLowDimLidarRangesVelsObss[:, 1:])
            robot2LidarClusteringStrgy =  PosVelsClusteringStrgy(lidarClustersNum
                                                                 , robot2TimeLowDimLidarRangesVelsObss[:, 1:])
            #two alphabet trans matrix building
            lidarTwoAlphWordsTransMtx = TwoAlphabetWordsTransitionMatrix(robot1LidarClusteringStrgy
                                                                         ,robot2LidarClusteringStrgy
                                                                         ,robot1TimeLowDimLidarRangesVelsObss
                                                                         ,robot2TimeLowDimLidarRangesVelsObss
                                                                         )
            lidarTwoAlphWordsTransMtx.save(pathToLidarTwoAlphTransMtxFile)
        else:
            lidarTwoAlphWordsTransMtx = TwoAlphabetWordsTransitionMatrix.load(pathToLidarTwoAlphTransMtxFile)
    else:
        gpsTwoAlphWordsTransMtx = TwoAlphabetWordsTransitionMatrix.load(pathToGpsTwoAlphTransMtxFile)
        lidarTwoAlphWordsTransMtx = TwoAlphabetWordsTransitionMatrix.load(pathToLidarTwoAlphTransMtxFile)



    # From here we gather data for gps location and  lidar abnormalities and gps abnormalities
    testScenarioName = configs["testScenarioName"]
    pathToTestScenrio = basePath + "{}-scenario/".format(testScenarioName)
    pathToTestScenarioYamlFile = pathToTestScenrio + "uav1-gps-lidar-uav2-gps-lidar.yaml"

    #Lidar settings
    lidarTestScenarioCounterLimit = configs["lidarTestScenarioCounterLimit"]
    robot1LidarTopicCounter = 0
    robot2LidarTopicCounter = 0

    robot1LidarTimeLowDimRangesObss = []
    robot2LidarTimeLowDimRangesObss = []
    robot1LidarTimeLowDimRangesVelsObss = np.empty([1, 2 * lidarAutoencoderLatentDim + 1])
    robot2LidarTimeLowDimRangesVelsObss = np.empty([1, 2 * lidarAutoencoderLatentDim + 1])
    lidarTimeAbnormalityValues = []
    lidarNoiseCovMtx = np.array(configs["rplidar"]["covMtx"])
    lidarTwoALphWordsClusterLevelAbnVals = TwoAlphabetWordsClusterLevelAbnormalVals(lidarTwoAlphWordsTransMtx)


    #GPS settings
    targetRobotGpsTimeRows = []
    robot1GpsCounter = 0
    robot2GpsCounter = 0
    robot1GpsTimeVelsObss = []
    robot2GpsTimeRangesVelsObss = np.empty([1, gpsVelsDim + 1])
    gpsTimeAbnormalityValues = []
    gpsCovMtx = np.array(configs["gps_origin"]["covMtx"])
    gpsTwoALphWordsClusterLevelAbnVals = TwoAlphabetWordsClusterLevelAbnormalVals(gpsTwoAlphWordsTransMtx)

    #plotting
    plotAll = PlotAll()

    with open(pathToTestScenarioYamlFile, "r") as file:
        topicRows = yaml.load_all(file, Loader=CLoader)
        targetRobotLidarTopicRowCounter = 0

        # load encoder
        robot1LidarEncoder = Autoencoder.loadEncoder(pathToRobot1LidarMindDir + "encoder.h5")
        robot2LidarEncoder = Autoencoder.loadEncoder(pathToRobot2LidarMindDir + "encoder.h5")

        #loop through topics
        for topicRowCounter, topicRow in enumerate(topicRows):
            if robot1LidarTopicCounter >= lidarTestScenarioCounterLimit:
                break

            robotId, sensorName = Topic.staticGetRobotIdAndSensorName(topicRow)
            time = Topic.staticGetTimeByTopicDict(topicRow)


            if sensorName == "gps_origin":
                gpsX,gpsY,gpsZ = GpsOrigin.staticGetXyz(topicRow)

                if robotId == configs["targetRobotIds"][0]:
                    if robot1GpsCounter == 0:
                        robot1GpsTimeXyzVelsObss = np.asarray([[time, gpsX, gpsY, gpsZ, 0, 0, 0]])
                        robot1GpsCounter += 1
                        continue
                    if robot1GpsCounter >= 1:
                        robot1GpsTimeXyzVelsObss = np.vstack((robot1GpsTimeXyzVelsObss, [time, gpsX, gpsY, gpsZ, 0, 0, 0]))
                        robot1GpsTimeDiff = robot1GpsTimeXyzVelsObss[-1][0] - robot1GpsTimeXyzVelsObss[-2][0]
                        robot1GpsDiff = np.subtract(robot1GpsTimeXyzVelsObss[-1][1:int(gpsVelsDim / 2)], robot1GpsTimeXyzVelsObss[-2][1:int(gpsVelsDim / 2)])
                        robot1GpsVels = gpsVelCo*robot1GpsDiff/robot1GpsTimeDiff
                        robot1GpsTimeXyzVelsObss[-1][int(gpsVelsDim / 2) + 1:gpsVelsDim]=robot1GpsVels
                        robot1GpsCounter += 1
                        
                if robotId == configs["targetRobotIds"][1]:
                    if robot2GpsCounter == 0:
                        robot2GpsTimeXyzVelsObss = np.asarray([[time, gpsX, gpsY, gpsZ, 0, 0, 0]])
                        robot2GpsCounter += 1
                        continue
                    if robot2GpsCounter >= 1:
                        robot2GpsTimeXyzVelsObss = np.vstack((robot2GpsTimeXyzVelsObss, [time, gpsX, gpsY, gpsZ, 0, 0, 0]))
                        robot2GpsTimeDiff = robot2GpsTimeXyzVelsObss[-1][0] - robot2GpsTimeXyzVelsObss[-2][0]
                        robot2GpsDiff = np.subtract(robot2GpsTimeXyzVelsObss[-1][1:int(gpsVelsDim / 2)], robot2GpsTimeXyzVelsObss[-2][1:int(gpsVelsDim / 2)])
                        robot2GpsVels = gpsVelCo*robot2GpsDiff/robot2GpsTimeDiff
                        robot2GpsTimeXyzVelsObss[-1][int(gpsVelsDim / 2) + 1:gpsVelsDim]=robot2GpsVels
                        robot2GpsCounter += 1

                plotAll.updateGpsPlot(np.asarray(robot1GpsTimeXyzVelsObss))

                if not (robot1GpsCounter>=2 and robot2GpsCounter>=2):
                    continue


                # gps abn computer
                robot1GpsCurObs = robot1GpsTimeXyzVelsObss[-1, 1:]
                robot1GpsPrvObs = robot1GpsTimeXyzVelsObss[-2, 1:]

                robot2GpsCurObs = robot1GpsTimeXyzVelsObss[-1, 1:]
                robot2GpsPrvObs = robot1GpsTimeXyzVelsObss[-2, 1:]

                gpsAbnormalityValue = gpsTwoALphWordsClusterLevelAbnVals.getCurAbnormalValByPrvAndCurPosVelObs(
                    robot1GpsPrvObs
                     ,robot2GpsPrvObs
                     ,robot1GpsCurObs
                     ,robot2GpsCurObs
                     )
                gpsTimeAbnormalityValues.append([time, gpsAbnormalityValue])
                plotAll.updateGpsAbnPlot(np.array(gpsTimeAbnormalityValues))


            if sensorName == "rplidar":
                npRanges = RpLidar.staticGetNpRanges(topicRow)

                if robotId == configs["targetRobotIds"][0]:
                    robot1LowDimLidarObs = robot1LidarEncoder(np.asarray([npRanges]))[0]
                    robot1LidarTimeLowDimRangesObss.append(np.insert(robot1LowDimLidarObs, 0, time, axis=0))
                    if robot1LidarTopicCounter == 0:
                        robot1LidarTopicCounter += 1
                        continue
                    if robot1LidarTopicCounter >= 1:
                        prvRobot1LidarTime = robot1LidarTimeLowDimRangesObss[robot1LidarTopicCounter - 1][0]
                        curRobot1LidarTime = robot1LidarTimeLowDimRangesObss[robot1LidarTopicCounter][0]
                        diffRobot1LidarTime = curRobot1LidarTime - prvRobot1LidarTime

                        prvRobot1LidarRanges = robot1LidarTimeLowDimRangesObss[robot1LidarTopicCounter - 1][1:]
                        curRobot1LidarRanges = robot1LidarTimeLowDimRangesObss[robot1LidarTopicCounter][1:]
                        diffRobot1LidarRanges = np.subtract(curRobot1LidarRanges, prvRobot1LidarRanges)

                        curRobot1LidarVel = lidarVelCo * diffRobot1LidarRanges / diffRobot1LidarTime
                        curRobot1LidarTimeRangesVels = np.hstack(np.array([curRobot1LidarTime, curRobot1LidarRanges, curRobot1LidarVel], dtype=object))
                        robot1LidarTimeLowDimRangesVelsObss =np.vstack([robot1LidarTimeLowDimRangesVelsObss, curRobot1LidarTimeRangesVels])
                    if robot1LidarTopicCounter == 1:
                        robot1LidarTimeLowDimRangesVelsObss = np.delete(robot1LidarTimeLowDimRangesVelsObss, 0, 0)
                        robot1LiarRangesVelsToAdd = np.hstack(np.array([prvRobot1LidarTime
                                                                           , prvRobot1LidarRanges
                                                                           , curRobot1LidarVel]
                                                                       , dtype=object))
                        robot1LidarTimeLowDimRangesVelsObss= np.insert(robot1LidarTimeLowDimRangesVelsObss, 0, robot1LiarRangesVelsToAdd, axis=0)

                    robot1LidarTimeLowDimRangesVelsObss = np.array(robot1LidarTimeLowDimRangesVelsObss)
                    robot1LidarTopicCounter += 1
                    
                if robotId == configs["targetRobotIds"][1]:
                    robot2LowDimLidarObs = robot2LidarEncoder(np.asarray([npRanges]))[0]
                    robot2LidarTimeLowDimRangesObss.append(np.insert(robot2LowDimLidarObs, 0, time, axis=0))
                    if robot2LidarTopicCounter == 0:
                        robot2LidarTopicCounter += 1
                        continue
                    if robot2LidarTopicCounter >= 1:
                        prvRobot1LidarTime = robot2LidarTimeLowDimRangesObss[robot2LidarTopicCounter - 1][0]
                        curRobot1LidarTime = robot2LidarTimeLowDimRangesObss[robot2LidarTopicCounter][0]
                        diffRobot1LidarTime = curRobot1LidarTime - prvRobot1LidarTime

                        prvRobot1LidarRanges = robot2LidarTimeLowDimRangesObss[robot2LidarTopicCounter - 1][1:]
                        curRobot1LidarRanges = robot2LidarTimeLowDimRangesObss[robot2LidarTopicCounter][1:]
                        diffRobot1LidarRanges = np.subtract(curRobot1LidarRanges, prvRobot1LidarRanges)

                        curRobot1LidarVel = lidarVelCo * diffRobot1LidarRanges / diffRobot1LidarTime
                        curRobot1LidarTimeRangesVels = np.hstack(np.array([curRobot1LidarTime, curRobot1LidarRanges, curRobot1LidarVel], dtype=object))
                        robot2LidarTimeLowDimRangesVelsObss =np.vstack([robot2LidarTimeLowDimRangesVelsObss, curRobot1LidarTimeRangesVels])
                    if robot2LidarTopicCounter == 1:
                        robot2LidarTimeLowDimRangesVelsObss = np.delete(robot2LidarTimeLowDimRangesVelsObss, 0, 0)
                        robot2LiarRangesVelsToAdd = np.hstack(np.array([prvRobot1LidarTime
                                                                           , prvRobot1LidarRanges
                                                                           , curRobot1LidarVel]
                                                                       , dtype=object))
                        robot2LidarTimeLowDimRangesVelsObss= np.insert(robot2LidarTimeLowDimRangesVelsObss, 0, robot2LiarRangesVelsToAdd, axis=0)

                    robot2LidarTimeLowDimRangesVelsObss = np.array(robot2LidarTimeLowDimRangesVelsObss)
                    robot2LidarTopicCounter += 1


                if not (robot1LidarTopicCounter>=2 and robot2LidarTopicCounter>=2):
                    continue

                # lidar abn computer
                robot1LidarCurObs = robot1LidarTimeLowDimRangesVelsObss[-1, 1:]
                robot1LidarPrvObs = robot1LidarTimeLowDimRangesVelsObss[-2, 1:]

                robot2LidarCurObs = robot1LidarTimeLowDimRangesVelsObss[-1, 1:]
                robot2LidarPrvObs = robot1LidarTimeLowDimRangesVelsObss[-2, 1:]

                lidarAbnormalityValue = lidarTwoALphWordsClusterLevelAbnVals.getCurAbnormalValByPrvAndCurPosVelObs(
                    robot1LidarPrvObs
                    , robot2LidarPrvObs
                    , robot1LidarCurObs
                    , robot2LidarCurObs
                )
                lidarTimeAbnormalityValues.append([time, lidarAbnormalityValue])
                plotAll.updateLidarAbnPlot(np.asarray(lidarTimeAbnormalityValues))
