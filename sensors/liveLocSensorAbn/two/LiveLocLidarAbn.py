import os
import pickle
import numpy as np
from MachineSettings import MachineSettings
from ctumrs.OneAlphabetWordsTransitionMatrix import OneAlphabetWordsTransitionMatrix
from ctumrs.TimePosVelsClusteringStrgy import TimePosVelsClusteringStrgy
import yaml
from yaml import CLoader

from ctumrs.TwoAlphabetWordsTransitionMatrix import TwoAlphabetWordsTransitionMatrix
from ctumrs.sensors.lidar.Autoencoder import Autoencoder
from ctumrs.sensors.liveLocSensorAbn.one.PlotAll import PlotAll
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
    testScenarioName = configs["testScenarioName"]
    basePath = MachineSettings.MAIN_PATH+"projs/research/data/self-aware-drones/ctumrs/two-drones/"
    pathToNormalScenario = basePath+"{}-scenario/".format(normalScenarioName)
    pathToNormalScenarioYamlFile = pathToNormalScenario+"uav1-gps-lidar-uav2-gps-lidar.yaml"


    #Lidar settings
    lidarSensorName = "rplidar"
    lidarClustersNum = configs["rplidar"]["clustersNum"]
    lidarVelCo = configs["rplidar"]["velCo"]
    lidarAutoencoderLatentDim = configs["rplidar"]["autoencoder"]["latentDim"]
    lidarAutoencoderEpochs = configs["rplidar"]["autoencoder"]["epocs"]
    lidarAutoencoderBatchSize = configs["rplidar"]["autoencoder"]["batchSize"]
    lidarTrainingRowsNumLimit = configs["rplidar"]["autoencoder"]["trainingRowsNumLimit"]

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

    pathToLidarTwoAlphaTransMtxFile = pathToLidarTwoAlphaTransMtxDir+"transMtx_training_{}_velco_{}_clusters_{}_autoencoder_latentdim_{}_epochs_{}.pk".format(
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
    gpsTrainingRowsNumLimit = configs["gps_origin"]["trainingRowsNumLimit"]
    gpsVelsDim = 6
    pathToGpsTwoAlphaTransMtxDir = pathToNormalScenario + gpsSensorName + "/"
    pathToGpsTwoAlphTransMtxFile = pathToGpsTwoAlphaTransMtxDir + "training_{}_velco_{}_clusters_{}/".format(
        gpsTrainingRowsNumLimit
         , gpsVelCo
         , gpsClustersNum
         )

    # normal scenrio lidar and gps trans matrix builder if not exist
    if TopicLoopingLogic.getShouldLoopThroughTopics(pathToGpsTwoAlphTransMtxFile,pathToLidarTwoAlphaTransMtxFile):
        with open(pathToNormalScenarioYamlFile, "r") as file:
            topicRows = yaml.load_all(file, Loader=CLoader)

            robot1TimeLowDimLidarRangesVelsObss = []
            robot2TimeLowDimLidarRangesVelsObss = []
            robot1TimeGpsVelsObss = []
            robot2TimeGpsVelsObss = []
            targetRobotLidarTopicRowCounter = 0
            targetRobotsGpsTopicRowCounter = 0

            for topicRowCounter, topicRow in enumerate(topicRows):
                if (targetRobotLidarTopicRowCounter >= lidarTrainingRowsNumLimit or os.path.exists(pathToRobot1LidarMindDir)) \
                        and (targetRobotsGpsTopicRowCounter >= gpsTrainingRowsNumLimit or os.path.exists(pathToTargetRobotGpsMindDir)):
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
                    if not os.path.exists(pathToRobot1LidarMindDir):
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
            robot1GpsVelsObssClusteringStrgy = TimePosVelsClusteringStrgy(gpsClustersNum
                                                                          , robot1TimeGpsVelsObss[:, 1:gpsVelsDim + 1])

            robot2TimeGpsVelsObss = np.asarray(robot2TimeGpsVelsObss)
            robot2TimeGpsVelsObss = TimePosRowsDerivativeComputer.computer(robot2TimeGpsVelsObss,gpsVelCo)
            robot2GpsVelsObssClusteringStrgy = TimePosVelsClusteringStrgy(gpsClustersNum
                                                                          , robot2TimeGpsVelsObss[:,1:gpsVelsDim + 1])

            gpsTwoAlphWordsTransMtx = TwoAlphabetWordsTransitionMatrix(robot1GpsVelsObssClusteringStrgy
                                                                         , robot2GpsVelsObssClusteringStrgy
                                                                         , robot1TimeGpsVelsObss[:, 1:gpsVelsDim + 1]
                                                                         , robot2TimeGpsVelsObss[:, 1:gpsVelsDim + 1]
                                                                         )

        else:
            with open(pathToGpsTwoAlphTransMtxFile) as file:
                gpsTwoAlphWordsTransMtx = pickle.load(file)

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
            robot1TimeLowDimLidarRangesVelsObss = TimePosRowsDerivativeComputer.computer(robot1LidarLowDimTimeObss
                                                                                         , lidarVelCo)

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
            robot2TimeLowDimLidarRangesVelsObss = TimePosRowsDerivativeComputer.computer(robot2LidarLowDimTimeObss,
                                                                                         lidarVelCo)
        if not os.path.exists(pathToLidarTwoAlphaTransMtxFile):
            #cluster each
            robot1LidarClusteringStrgy =  TimePosVelsClusteringStrgy(lidarClustersNum, robot1TimeLowDimLidarRangesVelsObss)
            robot2LidarClusteringStrgy =  TimePosVelsClusteringStrgy(lidarClustersNum, robot2TimeLowDimLidarRangesVelsObss)
            #two alphabet trans matrix building
            lidarTwoAlphWordsTransMtx = TwoAlphabetWordsTransitionMatrix(robot1LidarClusteringStrgy
                                                                         ,robot2LidarClusteringStrgy
                                                                         ,robot1TimeLowDimLidarRangesVelsObss
                                                                         ,robot2TimeLowDimLidarRangesVelsObss
                                                                         )
        else:
            lidarTwoAlphWordsTransMtx = TwoAlphabetWordsTransitionMatrix.load(pathToLidarTwoAlphaTransMtxFile)
    else:
        with open(pathToGpsTwoAlphTransMtxFile, 'rb') as file:
            gpsTwoAlphWordsTransMtx = pickle.load(file)
        with open(pathToLidarTwoAlphaTransMtxFile, 'rb') as file:
            lidarTwoAlphWordsTransMtx = pickle.load(file)



    # From here we gather data for gps location and  lidar abnormalities and gps abnormalities
    pathToTestScenrio = basePath + "{}-scenario/".format(testScenarioName)
    pathToTestScenarioYamlFile = pathToTestScenrio + "uav1-gps-lidar-uav2-gps-lidar.yaml"

    #Lidar settings
    targetRobotLidarCounterLimit = 6000
    targetRobotLidarCounter = 0
    targetRobotLidarTimeLowDimRangesObss = []
    targetRobotLidarTimeLowDimRangesVelsObss = np.empty([1, 2 * lidarAutoencoderLatentDim + 1])
    lidarTimeAbnormalityValues = []
    lidarNoiseCovVal = 0.01**2
    lidarNoiseCovMtx = np.array([
        [lidarNoiseCovVal, 0, 0, 0, 0, 0]
      , [0, lidarNoiseCovVal, 0, 0, 0, 0]
      , [0, 0, lidarNoiseCovVal, 0, 0, 0]
      , [0, 0, 0, lidarNoiseCovVal, 0, 0]
      , [0, 0, 0, 0, lidarNoiseCovVal, 0]
      , [0, 0, 0, 0, 0, lidarNoiseCovVal]
        ])

    #GPS settings
    targetRobotGpsTimeRows = []
    targetRobotGpsCounter = 0
    targetRobotGpsTimeVelsObss = []
    targetRobotGpsTimeRangesVelsObss = np.empty([1, gpsVelsDim + 1])
    gpsTimeAbnormalityValues = []
    gpsCovValPosXy = 2.0e-4
    gpsCovValPosZz = 4.0e-4
    gpsCovValVelXy = 0.2
    gpsCovValVelZz = 0.4
    gpsCovMtx = np.array([
        [gpsCovValPosXy, 0, 0, 0, 0, 0]
      , [0, gpsCovValPosXy, 0, 0, 0, 0]
      , [0, 0, gpsCovValPosZz, 0, 0, 0]
      , [0, 0, 0, gpsCovValVelXy, 0, 0]
      , [0, 0, 0, 0, gpsCovValVelXy, 0]
      , [0, 0, 0, 0, 0, gpsCovValVelZz]
        ])




    plotAll = PlotAll()

    with open(pathToTestScenarioYamlFile, "r") as file:
        topicRows = yaml.load_all(file, Loader=CLoader)
        targetRobotLidarTopicRowCounter = 0

        # load encoder
        encoder = Autoencoder.loadEncoder(pathToRobot1LidarMindDir + "encoder.h5")
        for topicRowCounter, topicRow in enumerate(topicRows):
            if targetRobotLidarCounter >= targetRobotLidarCounterLimit:
                break

            robotId, sensorName = Topic.staticGetRobotIdAndSensorName(topicRow)
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
                    diffGps = np.subtract(targetRobotGpsTimeXyzVelsObss[-1][1:int(gpsVelsDim/2)],targetRobotGpsTimeXyzVelsObss[-2][1:int(gpsVelsDim/2)])
                    gpsVels = gpsVelCo*diffGps/timeDiff
                    targetRobotGpsTimeXyzVelsObss[-1][int(gpsVelsDim/2)+1:gpsVelsDim]=gpsVels

                plotAll.updateGpsPlot(np.asarray(targetRobotGpsTimeXyzVelsObss))
                targetRobotGpsCounter += 1

                # gps abn computer
                gpsCurObs = targetRobotGpsTimeXyzVelsObss[-1, 1:]
                gpsPrvObs = targetRobotGpsTimeXyzVelsObss[-2, 1:]
                gpsPrvObsLabel = gpsTwoAlphWordsTransMtx.getClusteringStrgy().getPredictedLabelByPosVelObs(
                    gpsPrvObs)
                gpsPredictedNextLabel = gpsTwoAlphWordsTransMtx.getHighestPorobabelNextLabelBasedOnthePrvOne(
                    gpsPrvObsLabel)
                gpsPredictedNextLabelCenter = gpsTwoAlphWordsTransMtx.getClusteringStrgy().getClusterCenterByLabel(
                    gpsPredictedNextLabel)
                # gpsAbnormalityValue = np.linalg.norm(np.array(gpsCurObs) - np.array(gpsPredictedNextLabelCenter))
                gpsAbnormalityValue = Distance.getGaussianKullbackLieblerDistance(gpsCurObs
                                                                                  ,gpsCovMtx
                                                                                  ,gpsPredictedNextLabelCenter
                                                                                  ,gpsCovMtx)
                print("Gps abnormality value: " + str(gpsAbnormalityValue))
                gpsTimeAbnormalityValues.append([time, gpsAbnormalityValue])

                plotAll.updateGpsAbnPlot(np.array(gpsTimeAbnormalityValues))


            elif sensorName == "rplidar":
                npRanges = RpLidar.staticGetNpRanges(topicRow)
                lowDimLidarObs = encoder(np.asarray([npRanges]))[0]
                targetRobotLidarTimeLowDimRangesObss.append(np.insert(lowDimLidarObs, 0, time, axis=0))
                if targetRobotLidarCounter == 0:
                    targetRobotLidarCounter += 1
                    continue
                if targetRobotLidarCounter >= 1:
                    prvLidarTime = targetRobotLidarTimeLowDimRangesObss[targetRobotLidarCounter - 1][0]
                    curLidarTime = targetRobotLidarTimeLowDimRangesObss[targetRobotLidarCounter][0]
                    diffLidarTime = curLidarTime - prvLidarTime

                    prvLidarRanges = targetRobotLidarTimeLowDimRangesObss[targetRobotLidarCounter - 1][1:]
                    curLidarRanges = targetRobotLidarTimeLowDimRangesObss[targetRobotLidarCounter][1:]
                    diffLidarRanges = np.subtract(curLidarRanges, prvLidarRanges)

                    curLidarVel = lidarVelCo * diffLidarRanges / diffLidarTime
                    curLidarTimeRangesVels = np.hstack(np.array([curLidarTime, curLidarRanges, curLidarVel], dtype=object))
                    targetRobotLidarTimeLowDimRangesVelsObss =np.vstack([targetRobotLidarTimeLowDimRangesVelsObss, curLidarTimeRangesVels])
                if targetRobotLidarCounter == 1:
                    targetRobotLidarTimeLowDimRangesVelsObss = np.delete(targetRobotLidarTimeLowDimRangesVelsObss, 0, 0)
                    liarRangesVelsToAdd = np.hstack(np.array([prvLidarTime, prvLidarRanges, curLidarVel], dtype=object))
                    targetRobotLidarTimeLowDimRangesVelsObss= np.insert(targetRobotLidarTimeLowDimRangesVelsObss, 0, liarRangesVelsToAdd, axis=0)

                targetRobotLidarTimeLowDimRangesVelsObss = np.array(targetRobotLidarTimeLowDimRangesVelsObss)

                # lidar abn computer
                lidarCurObs = targetRobotLidarTimeLowDimRangesVelsObss[-1, 1:]
                lidarPrvObs = targetRobotLidarTimeLowDimRangesVelsObss[-2, 1:]
                lidarPrvObsLabel = lidarOneAlphabetWordsTransitionMatrix.getClusteringStrgy().getPredictedLabelByPosVelObs(lidarPrvObs)
                lidarPredictedNextLabel = lidarOneAlphabetWordsTransitionMatrix.getHighestPorobabelNextLabelBasedOnthePrvOne(
                    lidarPrvObsLabel)
                lidarPredictedNextLabelCenter = lidarOneAlphabetWordsTransitionMatrix.getClusteringStrgy().getClusterCenterByLabel(
                    lidarPredictedNextLabel)
                # lidarAbnormalityValue = np.linalg.norm(np.array(lidarCurObs) - np.array(lidarPredictedNextLabelCenter))
                lidarAbnormalityValue = Distance.getGaussianKullbackLieblerDistance(lidarCurObs, lidarNoiseCovMtx, lidarPredictedNextLabelCenter, lidarNoiseCovMtx)
                print("Lidar abnormality value: " + str(lidarAbnormalityValue))
                lidarTimeAbnormalityValues.append([time, lidarAbnormalityValue])

                plotAll.updateLidarAbnPlot(np.array(lidarTimeAbnormalityValues))
                targetRobotLidarCounter += 1