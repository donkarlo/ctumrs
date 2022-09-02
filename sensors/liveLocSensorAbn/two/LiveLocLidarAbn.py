import os
import pickle

import numpy as np

from MachineSettings import MachineSettings
from ctumrs.OneAlphabetWordsTransitionMatrix import OneAlphabetWordsTransitionMatrix
from ctumrs.TimePosVelsClusteringStrgy import TimePosVelsClusteringStrgy
import yaml
from yaml import CLoader

from ctumrs.sensors.liveLocSensorAbn.one.PlotAll import PlotAll
from ctumrs.topic.GpsOrigin import GpsOrigin
from ctumrs.topic.RpLidar import RpLidar
from ctumrs.topic.Topic import Topic
from mMath.calculus.derivative.TimePosRowsDerivativeComputer import TimePosRowsDerivativeComputer
from tensorflow.keras.models import Model, Sequential,load_model
from tensorflow.keras.layers import Dense

from mMath.statistic.Distance import Distance

if __name__ == "__main__":
    #the settings
    targetRobot1Id = "uav1"
    targetRobot2Id = "uav2"
    normalScenarioName = "normal"
    testScenarioName = "follow"
    basePath = MachineSettings.MAIN_PATH+"projs/research/data/self-aware-drones/ctumrs/two-drones/"
    pathToNormalScenario = basePath+"{}-scenario/".format(normalScenarioName)
    pathToNormalScenarioYamlFile = pathToNormalScenario+"uav1-gps-lidar-uav2-gps-lidar.yaml"


    #Lidar settings
    lidarSensorName = "rplidar"
    lidarClustersNum = 75
    lidarVelCo = 25
    lidarRangesDim = 720
    lidarAutoencoderLatentDim = 3
    lidarAutoencoderEpochs = 100
    lidarAutoencoderBatchSize = 32
    targetRobotLidarTopicRowLimit = 6000

    lidarRangesVelsDim = lidarRangesDim * 2
    pathToTargetRobotLidarMindDir = pathToNormalScenario + "{}/{}_mind_training_{}_velco_{}_clusters_{}_autoencoder_latentdim_{}_epochs_{}/".format(
        targetRobotId
        , lidarSensorName
        , targetRobotLidarTopicRowLimit
        , lidarVelCo
        , lidarClustersNum
        , lidarAutoencoderLatentDim
        , lidarAutoencoderEpochs
    )

    #Gps settings
    gpsSensorName = "gps_origin"
    gpsVelCo = 25
    gpsPosDim = 3
    gpsClustersNum = 75
    gpsUpdateRate = 0.01
    targetRobotGpsTopicRowLimit = 200000

    gpsPosVelsDim = gpsPosDim * 2
    pathToTargetRobotGpsMindDir = pathToNormalScenario + "{}/{}_mind_training_{}_velco_{}_clusters_{}/".format(
        targetRobotId
        , gpsSensorName
        , targetRobotGpsTopicRowLimit
        , gpsVelCo
        , gpsClustersNum
    )


    # normal scenrio lidar and gps trans matrix builder if not exist
    if not os.path.exists(pathToTargetRobotLidarMindDir) or not os.path.exists(pathToTargetRobotGpsMindDir):

        with open(pathToNormalScenarioYamlFile, "r") as file:
            topicRows = yaml.load_all(file, Loader=CLoader)

            targetRobotTimeLidarRangesVelsObss = []
            robotTimeGpsVelsObss = []
            targetRobotLidarTopicRowCounter = 0
            targetRobotGpsTopicRowCounter = 0

            for topicRowCounter, topicRow in enumerate(topicRows):
                if targetRobotLidarTopicRowCounter >= targetRobotLidarTopicRowLimit and targetRobotGpsTopicRowCounter>=targetRobotGpsTopicRowLimit:
                    break
                robotId, sensorName = Topic.staticGetRobotIdAndSensorName(topicRow)
                if robotId!=targetRobotId:
                    continue

                time = Topic.staticGetTimeByTopicDict(topicRow)

                if targetRobotLidarTopicRowCounter < targetRobotLidarTopicRowLimit:
                    if not os.path.exists(pathToTargetRobotLidarMindDir):
                        if sensorName==lidarSensorName:
                            npRanges = RpLidar.staticGetNpRanges(topicRow)
                            targetRobotTimeLidarRangesVelsObss.append(np.insert(npRanges, 0, time, axis=0))
                            targetRobotLidarTopicRowCounter += 1
                            print("robot: {}, Sensor: {}, count: {}".format(robotId, sensorName
                                                                            ,targetRobotLidarTopicRowCounter))
                if targetRobotGpsTopicRowCounter < targetRobotGpsTopicRowLimit:
                    if not os.path.exists(pathToTargetRobotGpsMindDir):
                        if sensorName==gpsSensorName:
                            gpsX,gpsY,gpsZ = GpsOrigin.staticGetXyz(topicRow)
                            robotTimeGpsVelsObss.append([time, gpsX, gpsY, gpsZ])
                            targetRobotGpsTopicRowCounter += 1
                            print("robot: {}, Sensor: {}, count: {}".format(robotId, sensorName,
                                                                            targetRobotGpsTopicRowCounter))

        # calculating transMtx for rplidar
        if not os.path.exists(pathToTargetRobotLidarMindDir):
            os.mkdir(pathToTargetRobotLidarMindDir)
            targetRobotTimeLidarRangesVelsObss = np.asarray(targetRobotTimeLidarRangesVelsObss)
            #normalize lidar data
            normalizedNpLeaderRangesObss = targetRobotTimeLidarRangesVelsObss[:,1:]
            #train the auto encoder
            encoder = Sequential([
                Dense(512, activation='relu', input_shape=(lidarRangesDim,)),
                Dense(256, activation='relu'),
                Dense(128, activation='relu'),
                Dense(64, activation='relu'),
                Dense(32, activation='relu'),
                Dense(lidarAutoencoderLatentDim, activation='relu')
            ])

            decoder = Sequential([
                Dense(32, activation='relu', input_shape=(lidarAutoencoderLatentDim,)),
                Dense(64, activation='relu'),
                Dense(128, activation='relu'),
                Dense(256, activation='relu'),
                Dense(512, activation='relu'),
                Dense(lidarRangesDim, activation=None)
            ])
            autoencoder = Model(inputs=encoder.input, outputs=decoder(encoder.output))
            autoencoder.compile(loss='mse', optimizer='adam')

            print("Fitting the auto encoder ...")
            modelHistory = autoencoder.fit(normalizedNpLeaderRangesObss
                                           , normalizedNpLeaderRangesObss
                                           , epochs=lidarAutoencoderEpochs
                                           , batch_size=lidarAutoencoderBatchSize
                                           , verbose=0)

            autoencoder.save(filepath=pathToTargetRobotLidarMindDir + "encoder-decoder.h5")
            decoder.save(filepath=pathToTargetRobotLidarMindDir + "decoder.h5")
            encoder.save(filepath=pathToTargetRobotLidarMindDir + "encoder.h5")

            print("Building the latent space of normal scenario ladar data ...")
            lidarLowDimObss = encoder.predict(normalizedNpLeaderRangesObss)
            lidarLowDimTimeObss = np.hstack((targetRobotTimeLidarRangesVelsObss[0:, 0:1],lidarLowDimObss))
            #compute velocities
            targetRobotTimeLidarRangesVelsObss = TimePosRowsDerivativeComputer.computer(lidarLowDimTimeObss, lidarVelCo)

            #clustering
            lidarRangesVelsObssClusteringStrgy = TimePosVelsClusteringStrgy(lidarClustersNum
                                                                            , targetRobotTimeLidarRangesVelsObss[:, 1:])
            #Building Lidar transition matrix
            lidarOneAlphabetWordsTransitionMatrix = OneAlphabetWordsTransitionMatrix(lidarRangesVelsObssClusteringStrgy
                                                                                     , targetRobotTimeLidarRangesVelsObss[:, 1:])
            lidarOneAlphabetWordsTransitionMatrix.getNpTransitionMatrix()
            lidarOneAlphabetWordsTransitionMatrix.save(pathToTargetRobotLidarMindDir+"transmtx.pkl")
        else:
            with open(pathToTargetRobotLidarMindDir+"transmtx.pkl", 'rb') as file:
                lidarOneAlphabetWordsTransitionMatrix = pickle.load(file)

        # calculating transMtx for GPS
        if not os.path.exists(pathToTargetRobotGpsMindDir):
            os.mkdir(pathToTargetRobotGpsMindDir)

            robotTimeGpsVelsObss = np.asarray(robotTimeGpsVelsObss)
            robotTimeGpsVelsObss = TimePosRowsDerivativeComputer.computer(robotTimeGpsVelsObss,
                                                                                  gpsVelCo,gpsUpdateRate)

            # clustering gps data
            gpsVelsObssClusteringStrgy = TimePosVelsClusteringStrgy(gpsClustersNum
                                                                    , robotTimeGpsVelsObss[:,
                                                                      1:gpsPosVelsDim + 1])
            gpsVelsObssClusteringStrgyDict = gpsVelsObssClusteringStrgy.getLabeledTimePosVelsClustersDict(
                robotTimeGpsVelsObss[:, 1:gpsPosVelsDim + 1])
            # Building Gps transition matrix
            gpsOneAlphabetWordsTransitionMatrix = OneAlphabetWordsTransitionMatrix(gpsVelsObssClusteringStrgy
                                                                                   , robotTimeGpsVelsObss[:,
                                                                                     1:gpsPosVelsDim + 1])
            gpsOneAlphabetWordsTransitionMatrix.getNpTransitionMatrix()


            gpsOneAlphabetWordsTransitionMatrix.save(pathToTargetRobotGpsMindDir+"transmtx.pkl")
        else:
            with open(pathToTargetRobotGpsMindDir+"transmtx.pkl", 'rb') as file:
                gpsOneAlphabetWordsTransitionMatrix = pickle.load(file)
    else:
        with open(pathToTargetRobotLidarMindDir+"transmtx.pkl", 'rb') as file:
            lidarOneAlphabetWordsTransitionMatrix = pickle.load(file)
        with open(pathToTargetRobotGpsMindDir+"transmtx.pkl", 'rb') as file:
            gpsOneAlphabetWordsTransitionMatrix = pickle.load(file)



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
    targetRobotGpsTimeRangesVelsObss = np.empty([1, gpsPosVelsDim + 1])
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
        encoder = load_model(pathToTargetRobotLidarMindDir + "encoder.h5")
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
                    if timeDiff == 0:
                        timeDiff = gpsUpdateRate
                    diffGps = np.subtract(targetRobotGpsTimeXyzVelsObss[-1][1:int(gpsPosVelsDim / 2)], targetRobotGpsTimeXyzVelsObss[-2][1:int(gpsPosVelsDim / 2)])
                    gpsVels = gpsVelCo*diffGps/timeDiff
                    targetRobotGpsTimeXyzVelsObss[-1][int(gpsPosVelsDim / 2) + 1:gpsPosVelsDim]=gpsVels

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