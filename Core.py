import pickle

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import silhouette_score
from tensorflow import keras
from tensorflow.keras.models import Model, Sequential,load_model
from tensorflow.keras.layers import Dense,LSTM


import os
from MachineSettings import MachineSettings
from ctumrs.PosVelsClusteringStrgy import PosVelsClusteringStrgy
import yaml
from yaml import CLoader
from ctumrs.TimePosVelObssPlottingUtility import TimePosVelObssPlottingUtility
from ctumrs.TwoAlphabetWordsClusterLevelAbnormalVals import TwoAlphabetWordsClusterLevelAbnormalVals
from ctumrs.TwoAlphabetWordsTransitionMatrix import TwoAlphabetWordsTransitionMatrix
from ctumrs.sensors.lidar.Autoencoder import Autoencoder
from ctumrs.sensors.liveLocSensorAbn.two.Abnormality import Abnormality
from ctumrs.sensors.liveLocSensorAbn.two.PlotPosGpsLidarLive import PlotPosGpsLidarLive
from ctumrs.sensors.liveLocSensorAbn.two.TopicLoopingLogic import TopicLoopingLogic
from ctumrs.sensors.noise.Noise import Noise
from ctumrs.topic.GpsOrigin import GpsOrigin
from ctumrs.topic.RpLidar import RpLidar
from ctumrs.topic.Topic import Topic
from mMath.calculus.derivative.TimePosRowsDerivativeComputer import TimePosRowsDerivativeComputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from yellowbrick.cluster import SilhouetteVisualizer

# configs
with open("configs.yaml", "r") as file:
    configs = yaml.load(file, Loader=CLoader)
testSharedPath = "/home/donkarlo/Desktop/lstm/"
numOfRobots = len(configs["targetRobotIds"])

gpsValVelDim = 6
lidarValVelDim = 1440
encodersLatentDim = 3
trainingSeqLen = 15
numOfCLusters = 20

class Core:
    def __init__(self):
        pass

    def translate(self):
        def transmit():
            pass
        def rotate():
            pass
        pass

    def encode(self):
        pass

    def buildTemporalModel(self):
        pass

    def recallTemporalModel(self):
        pass

    def pairing(self):
        pass

    def convertRecalledModelPredictionsToActions(self):
        pass

    def getTrainingSensoryData(self):
        targetRobotIds = configs["targetRobotIds"]
        normalScenarioName = configs["normalScenarioName"]
        basePath = MachineSettings.MAIN_PATH + "projs/research/data/self-aware-drones/ctumrs/two-drones/"
        pathToNormalScenario = basePath + "{}-scenario/".format(normalScenarioName)

        pathToNormalScenarioYamlFile = pathToNormalScenario + "uav1-gps-lidar-uav2-gps-lidar.yaml"

        # Lidar settings
        lidarSensorName = "rplidar"
        lidarTrainingRowsNumLimit = configs["rplidar"]["trainingRowsNumLimit"]
        lidarGaussianNoiseVar = configs["rplidar"]["gaussianNoiseVarCo"]
        robot1LidarEncoder = Autoencoder.loadEncoder(
            "/home/donkarlo/Dropbox/projs/research/data/self-aware-drones/ctumrs/two-drones/normal-scenario/uav1/rplidar_mind_gaussianNoiseVarCo_0_training_120000_velco_1_clusters_32_autoencoder_latentdim_3_epochs_200/" + "encoder.h5")
        robot2LidarEncoder = Autoencoder.loadEncoder(
            "/home/donkarlo/Dropbox/projs/research/data/self-aware-drones/ctumrs/two-drones/normal-scenario/uav2/rplidar_mind_gaussianNoiseVarCo_0_training_120000_velco_1_clusters_32_autoencoder_latentdim_3_epochs_200/" + "encoder.h5")

        # Gps settings
        gpsSensorName = "gps_origin"
        gpsTrainingRowsNumLimit = configs["gps_origin"]["trainingRowsNumLimit"]
        gpsGaussianNoiseVar = configs["gps_origin"]["gaussianNoiseVarCo"]

        if not os.path.exists("{}/normalRobot1TimeGpsVelsObss.pkl".format(testSharedPath)) \
                or not os.path.exists("{}/normalRobot2TimeGpsVelsObss.pkl".format(testSharedPath)) \
                or not os.path.exists("{}/normalRobot1TimeLowDimLidarRangesVelsObss.pkl".format(testSharedPath)) \
                or not os.path.exists("{}/normalRobot2TimeLowDimLidarRangesVelsObss.pkl".format(testSharedPath)) :
            with open(pathToNormalScenarioYamlFile, "r") as file:
                topicRows = yaml.load_all(file, Loader=CLoader)

                robot1TimeLowDimLidarRangesVelsObss = []
                robot2TimeLowDimLidarRangesVelsObss = []
                robot1TimeGpsVelsObss = []
                robot2TimeGpsVelsObss = []
                targetRobotLidarTopicRowCounter = 0
                targetRobotsGpsTopicRowCounter = 0

                for topicRowCounter, topicRow in enumerate(topicRows):
                    if (targetRobotLidarTopicRowCounter >= lidarTrainingRowsNumLimit) and (targetRobotsGpsTopicRowCounter >= gpsTrainingRowsNumLimit):
                        break
                    robotId, sensorName = Topic.staticGetRobotIdAndSensorName(topicRow)
                    time = Topic.staticGetTimeByTopicDict(topicRow)

                    if targetRobotsGpsTopicRowCounter < gpsTrainingRowsNumLimit:
                        if sensorName == gpsSensorName:
                            gpsX, gpsY, gpsZ = Noise.addSymetricGaussianNoiseToAVec(
                                np.array(GpsOrigin.staticGetGpsXyz(topicRow)), gpsGaussianNoiseVar)

                            if robotId == targetRobotIds[0]:
                                robot1TimeGpsVelsObss.append([time, gpsX, gpsY, gpsZ])
                            elif robotId == targetRobotIds[1]:
                                robot2TimeGpsVelsObss.append([time, gpsX, gpsY, gpsZ])

                            targetRobotsGpsTopicRowCounter += 1
                            print("robot: {}, Sensor: {}, count: {}".format(robotId
                                                                            , sensorName,
                                                                            targetRobotsGpsTopicRowCounter))

                    if targetRobotLidarTopicRowCounter < lidarTrainingRowsNumLimit:
                            if sensorName == lidarSensorName:
                                npRanges = Noise.addSymetricGaussianNoiseToAVec(RpLidar.staticGetNpRanges(topicRow),
                                                                                lidarGaussianNoiseVar)

                                if robotId == targetRobotIds[0]:
                                    npRobot1LowDimRanges = robot1LidarEncoder.predict(np.asarray([npRanges]))[0]
                                    robot1TimeLowDimLidarRangesVelsObss.append(np.insert(npRobot1LowDimRanges, 0, time, axis=0))
                                elif robotId == targetRobotIds[1]:
                                    npRobot2LowDimRanges = robot2LidarEncoder.predict(np.asarray([npRanges]))[0]
                                    robot2TimeLowDimLidarRangesVelsObss.append(np.insert(npRobot2LowDimRanges, 0, time, axis=0))

                                targetRobotLidarTopicRowCounter += 1
                                print("robot: {}, Sensor: {}, count: {}".format(robotId
                                                                                , sensorName
                                                                                , targetRobotLidarTopicRowCounter))

            robot1TimeGpsVelsObss = TimePosRowsDerivativeComputer.computer(robot1TimeGpsVelsObss, 1)
            robot2TimeGpsVelsObss = TimePosRowsDerivativeComputer.computer(robot2TimeGpsVelsObss, 1)
            with open('{}/normalRobot1TimeGpsVelsObss.pkl'.format(testSharedPath), 'wb') as file:
                pickle.dump(robot1TimeGpsVelsObss, file)
            with open('{}/normalRobot2TimeGpsVelsObss.pkl'.format(testSharedPath), 'wb') as file:
                pickle.dump(robot2TimeGpsVelsObss, file)
            with open('{}/normalRobot1TimeLowDimLidarRangesVelsObss.pkl'.format(testSharedPath), 'wb') as file:
                pickle.dump(robot1TimeLowDimLidarRangesVelsObss, file)
            with open('{}/normalRobot2TimeLowDimLidarRangesVelsObss.pkl'.format(testSharedPath), 'wb') as file:
                pickle.dump(robot2TimeLowDimLidarRangesVelsObss, file)
        else:
            with open('{}/normalRobot1TimeGpsVelsObss.pkl'.format(testSharedPath), 'rb') as file:
                robot1TimeGpsVelsObss =pickle.load(file)
            with open('{}/normalRobot2TimeGpsVelsObss.pkl'.format(testSharedPath), 'rb') as file:
                robot2TimeGpsVelsObss =pickle.load(file)
            with open('{}/normalRobot1TimeLowDimLidarRangesVelsObss.pkl'.format(testSharedPath), 'rb') as file:
                robot1TimeLowDimLidarRangesVelsObss =pickle.load(file)
            with open('{}/normalRobot2TimeLowDimLidarRangesVelsObss.pkl'.format(testSharedPath), 'rb') as file:
                robot2TimeLowDimLidarRangesVelsObss = pickle.load(file)


        combinedRobotsTimeGpsVelsObss = []
        for robot1TimeGpsVelsObsCounter,robot1TimeGpsVelsObs in enumerate(robot1TimeGpsVelsObss):
            if robot1TimeGpsVelsObsCounter>=len(robot2TimeGpsVelsObss):
                break
            robot2TimeGpsVelsObs = robot2TimeGpsVelsObss[robot1TimeGpsVelsObsCounter]
            combinedRobotsTimeGpsVelsObs = [robot1TimeGpsVelsObs,robot2TimeGpsVelsObs]
            combinedRobotsTimeGpsVelsObss.append(combinedRobotsTimeGpsVelsObs)
        npCombinedRobotsTimeGpsVelsObss = np.asarray(combinedRobotsTimeGpsVelsObss)

        robot1TimeLowDimLidarRangesVelsObss = TimePosRowsDerivativeComputer.computer(robot1TimeLowDimLidarRangesVelsObss, 1)
        robot2TimeLowDimLidarRangesVelsObss = TimePosRowsDerivativeComputer.computer(robot2TimeLowDimLidarRangesVelsObss, 1)
        combinedRobotsTimeLowDimLidarVelsObss = []
        for robot1TimeLowDimLidarVelsObsCounter, robot1TimeLowDimLidarVelsObs in enumerate(robot1TimeLowDimLidarRangesVelsObss):
            if robot1TimeLowDimLidarVelsObsCounter >= len(robot2TimeLowDimLidarRangesVelsObss):
                break
            robot2TimeLowDimLidarVelsObs = robot2TimeLowDimLidarRangesVelsObss[robot1TimeLowDimLidarVelsObsCounter]
            combinedRobotsTimeLowDimLidarVelsObs = [robot1TimeLowDimLidarVelsObs, robot2TimeLowDimLidarVelsObs]
            combinedRobotsTimeLowDimLidarVelsObss.append(combinedRobotsTimeLowDimLidarVelsObs)
        npCombinedRobotsTimeLowDimLidarVelsObss = np.asarray(combinedRobotsTimeLowDimLidarVelsObss)

        return npCombinedRobotsTimeGpsVelsObss,npCombinedRobotsTimeLowDimLidarVelsObss

    def getTrainingSequences(self, npRobotsEncodedObss:np.ndarray):
        '''

        Parameters
        ----------
        npRobotsEncodedObss
        trainingSeqsLen

        Returns
        -------
        A np.array of 1*12 matrices as inputSeqs
        '''
        #remove time from all matrices so that we have an array of 2*6 matrices

        inputSeqs = []
        relevantOutputSeqs = []

        for counter in range(len(npRobotsEncodedObss)):
            # get the last index
            lastIndex = counter + trainingSeqLen

            # if lastIndex is greater than length of sequence then break
            if lastIndex > len(npRobotsEncodedObss) - 1:
                break

            # Create input and output sequence
            inputSeq= npRobotsEncodedObss[counter:lastIndex]
            relevantOutputSeq = npRobotsEncodedObss[lastIndex]

            # append seq_X, seq_y in X and y list
            inputSeqs.append(inputSeq)
            relevantOutputSeqs.append(relevantOutputSeq)

        inputSeqs = np.array(inputSeqs)
        relevantOutputSeqs = np.array(relevantOutputSeqs)

        return inputSeqs, relevantOutputSeqs



    def getRelevantLstmAccordingToClosestCluster(self, lstmModelsDictByClusteringLabels:dict, clustering:KMeans, prvNormalizedReshapedPosVelsObssSeqLen:np.ndarray)->LSTM:
        predictionLabels = clustering.predict(prvNormalizedReshapedPosVelsObssSeqLen)
        mostFrequentLabel = np.bincount(predictionLabels).argmax()
        print(f"choosen cluster for the LSTM: {mostFrequentLabel}")
        return lstmModelsDictByClusteringLabels[mostFrequentLabel]


    def loopThroughTestSensoryData(self, gpsLstmModelsDictByClusteringLabels:dict,gpsClustering:KMeans,gpsEncoder,gpsScaler:StandardScaler):
        normalScenarioStartTime = configs["normalScenarioStartTime"]
        basePath = MachineSettings.MAIN_PATH + "projs/research/data/self-aware-drones/ctumrs/two-drones/"
        ############## From here we gather data for gps location and  lidar abnormalities and gps abnormalities
        testScenarioName = configs["testScenarioName"]
        pathToTestScenrio = basePath + "{}-scenario/".format(testScenarioName)
        pathToTestScenarioYamlFile = pathToTestScenrio + "uav1-gps-lidar-uav2-gps-lidar.yaml"

        # Lidar settings
        lidarTestScenarioCounterLimit = configs["lidarTestScenarioCounterLimit"]
        robot1LidarTopicCounter = 0

        # GPS settings
        robot1GpsCounter = 0
        robot2GpsCounter = 0
        gpsTimeAbnormalityValues = []
        robotsGpsEncodedPrvObss = []

        # Lidar settings
        lidarTestScenarioCounterLimit = configs["lidarTestScenarioCounterLimit"]
        robot1LidarTopicCounter = 0
        robot2LidarTopicCounter = 0

        robot1LidarTimeLowDimRangesObss = []
        robot2LidarTimeLowDimRangesObss = []
        robot1LowDimLidarPrvObssRobot2LowDimLidarPrvObss = []
        lidarAutoencoderLatentDim = configs["rplidar"]["autoencoder"]["latentDim"]
        robot1LidarTimeLowDimRangesVelsObss = np.empty([1, 2 * lidarAutoencoderLatentDim + 1])
        robot2LidarTimeLowDimRangesVelsObss = np.empty([1, 2 * lidarAutoencoderLatentDim + 1])
        lowDimLidarTimeAbnormalityValues = []
        # load encoder
        robot1LidarEncoder = Autoencoder.loadEncoder("/home/donkarlo/Dropbox/projs/research/data/self-aware-drones/ctumrs/two-drones/normal-scenario/uav1/rplidar_mind_gaussianNoiseVarCo_0_training_120000_velco_1_clusters_32_autoencoder_latentdim_3_epochs_200/" + "encoder.h5")
        robot2LidarEncoder = Autoencoder.loadEncoder("/home/donkarlo/Dropbox/projs/research/data/self-aware-drones/ctumrs/two-drones/normal-scenario/uav2/rplidar_mind_gaussianNoiseVarCo_0_training_120000_velco_1_clusters_32_autoencoder_latentdim_3_epochs_200/" + "encoder.h5")
        lidarGaussianNoiseVar = configs["rplidar"]["gaussianNoiseVarCo"]

        # plotting
        plotAll = PlotPosGpsLidarLive()
        with open('{}/followScenarioRobotIdTimeSensorObss.pkl'.format(testSharedPath), 'rb') as file:
            topicRows = pickle.load(file)

            prvTime = 0
            beginningSkipCounter = 0
            # loop through topics
            for topicRowCounter, topicRow in enumerate(topicRows):
                if beginningSkipCounter < configs["beginningSkip"]:
                    beginningSkipCounter += 1
                    continue
                if robot1LidarTopicCounter >= lidarTestScenarioCounterLimit:
                    break

                robotId, sensorName = topicRow["robotId"],topicRow["sensorName"]
                time = topicRow["time"] - normalScenarioStartTime
                if prvTime == 0:
                    prvTime = time

                if prvTime > time:
                    continue

                if sensorName == "gps_origin":
                    gpsX, gpsY, gpsZ = topicRow["npValue"]

                    if robotId == configs["targetRobotIds"][0]:
                        if robot1GpsCounter == 0:
                            robot1GpsTimeXyzVelsObss = np.asarray([[time, gpsX, gpsY, gpsZ, 0, 0, 0]])
                            robot1GpsCounter += 1
                            continue
                        if robot1GpsCounter >= 1:
                            robot1GpsTimeXyzVelsObss = np.vstack(
                                (robot1GpsTimeXyzVelsObss, [time, gpsX, gpsY, gpsZ, 0, 0, 0]))
                            robot1GpsTimeDiff = robot1GpsTimeXyzVelsObss[-1][0] - robot1GpsTimeXyzVelsObss[-2][0]
                            if robot1GpsTimeDiff == 0:
                                continue
                            robot1GpsDiff = np.subtract(robot1GpsTimeXyzVelsObss[-1][1:int(gpsValVelDim / 2)],
                                                        robot1GpsTimeXyzVelsObss[-2][1:int(gpsValVelDim / 2)])
                            robot1GpsVels = robot1GpsDiff / robot1GpsTimeDiff
                            robot1GpsTimeXyzVelsObss[-1][int(gpsValVelDim / 2) + 1:gpsValVelDim] = robot1GpsVels
                            robot1GpsCounter += 1

                    elif robotId == configs["targetRobotIds"][1]:
                        if robot2GpsCounter == 0:
                            robot2GpsTimeXyzVelsObss = np.asarray([[time, gpsX, gpsY, gpsZ, 0, 0, 0]])
                            robot2GpsCounter += 1
                            continue
                        if robot2GpsCounter >= 1:
                            robot2GpsTimeXyzVelsObss = np.vstack(
                                (robot2GpsTimeXyzVelsObss, [time, gpsX, gpsY, gpsZ, 0, 0, 0]))
                            robot2GpsTimeDiff = robot2GpsTimeXyzVelsObss[-1][0] - robot2GpsTimeXyzVelsObss[-2][0]
                            if robot2GpsTimeDiff == 0:
                                continue
                            robot2GpsDiff = np.subtract(robot2GpsTimeXyzVelsObss[-1][1:int(gpsValVelDim / 2)],
                                                        robot2GpsTimeXyzVelsObss[-2][1:int(gpsValVelDim / 2)])
                            robot2GpsVels = robot2GpsDiff / robot2GpsTimeDiff
                            robot2GpsTimeXyzVelsObss[-1][int(gpsValVelDim / 2) + 1:gpsValVelDim] = robot2GpsVels
                            robot2GpsCounter += 1

                    if topicRowCounter % configs["plotUpdateRate"] == 0:
                        plotAll.updateGpsPlot(np.asarray(robot1GpsTimeXyzVelsObss),
                                              np.asarray(robot2GpsTimeXyzVelsObss))

                    if robot1GpsCounter <= 1 or robot2GpsCounter <= 1:
                        continue

                    # gps abn computer
                    robot1GpsCurObs = robot1GpsTimeXyzVelsObss[-1, 1:]
                    robot2GpsCurObs = robot2GpsTimeXyzVelsObss[-1, 1:]
                    #convert to 1*12 array
                    robotsCurGpsFlattedObs = np.array([np.concatenate((robot1GpsCurObs, robot2GpsCurObs), axis=0)])
                    robotsGpsCurFlattedScaledObs = gpsScaler.transform(robotsCurGpsFlattedObs)
                    robotsGpsCurEncodedObs =  gpsEncoder.predict(robotsGpsCurFlattedScaledObs)


                    robot1GpsPrvObs = robot1GpsTimeXyzVelsObss[-2, 1:]
                    robot2GpsPrvObs = robot2GpsTimeXyzVelsObss[-2, 1:]
                    # convert to 1*encoder latent dim  array
                    robotsPrvGpsFlattedObs = np.array([np.concatenate((robot1GpsPrvObs, robot2GpsPrvObs), axis=0)])
                    robotsGpsPrvEncodedObs = gpsScaler.transform(robotsPrvGpsFlattedObs)
                    robotsGpsPrvEncodedObs = gpsEncoder.predict(robotsGpsPrvEncodedObs)
                    robotsGpsEncodedPrvObss.append(robotsGpsPrvEncodedObs)


                    if len(robotsGpsEncodedPrvObss)>trainingSeqLen:
                        #Take the last training seq len
                        robotsGpsEncodedObssLastTrainingSeqLen = np.asarray(robotsGpsEncodedPrvObss[-trainingSeqLen:])
                        #We must give predict an array of sequences, the following line converts prv to an array of such sequences shape (1,15,12) , lstm.predict needs an array of sequences
                        robotsGpsEncodedObssLastTrainingSeqLenReshaped = robotsGpsEncodedObssLastTrainingSeqLen.reshape((1,trainingSeqLen, encodersLatentDim))
                        bestGpsLstm = self.getRelevantLstmAccordingToClosestCluster(gpsLstmModelsDictByClusteringLabels,gpsClustering,robotsGpsEncodedObssLastTrainingSeqLenReshaped[0])
                        robotsGpsEncodedPrd = bestGpsLstm.predict(robotsGpsEncodedObssLastTrainingSeqLenReshaped)

                        gpsAbnormalityValue = np.linalg.norm(robotsGpsEncodedPrd - robotsGpsCurEncodedObs)
                        gpsTimeAbnormalityValues.append([time, gpsAbnormalityValue])
                        if topicRowCounter % configs["plotUpdateRate"] == 0:
                            plotAll.updateGpsAbnPlot(np.array(gpsTimeAbnormalityValues))
                # elif sensorName == "rplidar":
                #     npRanges = topicRow["npValue"]
                #
                #     if robotId == configs["targetRobotIds"][0]:
                #         robot1LowDimLidarObs = robot1LidarEncoder.predict((np.asarray([npRanges])))[0]
                #         robot1LidarTimeLowDimRangesObss.append(np.insert(robot1LowDimLidarObs, 0, time, axis=0))
                #         if robot1LidarTopicCounter == 0:
                #             robot1LidarTopicCounter += 1
                #             continue
                #         if robot1LidarTopicCounter >= 1:
                #             prvRobot1LidarTime = robot1LidarTimeLowDimRangesObss[robot1LidarTopicCounter - 1][0]
                #             curRobot1LidarTime = robot1LidarTimeLowDimRangesObss[robot1LidarTopicCounter][0]
                #             diffRobot1LidarTime = curRobot1LidarTime - prvRobot1LidarTime
                #             if diffRobot1LidarTime == 0:
                #                 continue
                #             prvRobot1LidarRanges = robot1LidarTimeLowDimRangesObss[robot1LidarTopicCounter - 1][1:]
                #             curRobot1LidarRanges = robot1LidarTimeLowDimRangesObss[robot1LidarTopicCounter][1:]
                #             diffRobot1LidarRanges = np.subtract(curRobot1LidarRanges, prvRobot1LidarRanges)
                #             curRobot1LidarVel = diffRobot1LidarRanges / diffRobot1LidarTime
                #             curRobot1LidarTimeRangesVels = np.hstack(
                #                 np.array([curRobot1LidarTime, curRobot1LidarRanges, curRobot1LidarVel], dtype=object))
                #             robot1LidarTimeLowDimRangesVelsObss = np.vstack(
                #                 [robot1LidarTimeLowDimRangesVelsObss, curRobot1LidarTimeRangesVels])
                #         if robot1LidarTopicCounter == 1:
                #             robot1LidarTimeLowDimRangesVelsObss = np.delete(robot1LidarTimeLowDimRangesVelsObss, 0, 0)
                #             robot1LidarRangesVelsToAdd = np.hstack(np.array([prvRobot1LidarTime
                #                                                                 , prvRobot1LidarRanges
                #                                                                 , curRobot1LidarVel]
                #                                                             , dtype=object))
                #             robot1LidarTimeLowDimRangesVelsObss = np.insert(robot1LidarTimeLowDimRangesVelsObss, 0,
                #                                                             robot1LidarRangesVelsToAdd, axis=0)
                #
                #         robot1LidarTimeLowDimRangesVelsObss = np.array(robot1LidarTimeLowDimRangesVelsObss)
                #         robot1LidarTopicCounter += 1
                #     elif robotId == configs["targetRobotIds"][1]:
                #         robot2LowDimLidarObs = robot2LidarEncoder.predict((np.asarray([npRanges])))[0]
                #         robot2LidarTimeLowDimRangesObss.append(np.insert(robot2LowDimLidarObs, 0, time, axis=0))
                #         if robot2LidarTopicCounter == 0:
                #             robot2LidarTopicCounter += 1
                #             continue
                #         if robot2LidarTopicCounter >= 1:
                #             prvRobot2LidarTime = robot2LidarTimeLowDimRangesObss[robot2LidarTopicCounter - 1][0]
                #             curRobot2LidarTime = robot2LidarTimeLowDimRangesObss[robot2LidarTopicCounter][0]
                #             diffRobot2LidarTime = curRobot2LidarTime - prvRobot2LidarTime
                #             if diffRobot2LidarTime == 0:
                #                 continue
                #             prvRobot2LidarRanges = robot2LidarTimeLowDimRangesObss[robot2LidarTopicCounter - 1][1:]
                #             curRobot2LidarRanges = robot2LidarTimeLowDimRangesObss[robot2LidarTopicCounter][1:]
                #             diffRobot2LidarRanges = np.subtract(curRobot2LidarRanges, prvRobot2LidarRanges)
                #             curRobot2LidarVel = diffRobot2LidarRanges / diffRobot2LidarTime
                #             curRobot2LidarTimeRangesVels = np.hstack(
                #                 np.array([curRobot2LidarTime, curRobot2LidarRanges, curRobot2LidarVel], dtype=object))
                #             robot2LidarTimeLowDimRangesVelsObss = np.vstack(
                #                 [robot2LidarTimeLowDimRangesVelsObss, curRobot2LidarTimeRangesVels])
                #         if robot2LidarTopicCounter == 1:
                #             robot2LidarTimeLowDimRangesVelsObss = np.delete(robot2LidarTimeLowDimRangesVelsObss, 0, 0)
                #             robot2LidarRangesVelsToAdd = np.hstack(np.array([prvRobot2LidarTime
                #                                                                 , prvRobot2LidarRanges
                #                                                                 , curRobot2LidarVel]
                #                                                             , dtype=object))
                #             robot2LidarTimeLowDimRangesVelsObss = np.insert(robot2LidarTimeLowDimRangesVelsObss, 0,
                #                                                             robot2LidarRangesVelsToAdd, axis=0)
                #
                #         robot2LidarTimeLowDimRangesVelsObss = np.array(robot2LidarTimeLowDimRangesVelsObss)
                #         robot2LidarTopicCounter += 1
                #     if robot1LidarTopicCounter <= 1 or robot2LidarTopicCounter <= 1:
                #         continue
                #
                #     # lowDimLidar abn computer
                #     robot1LowDimLidarCurObs = robot1LidarTimeLowDimRangesVelsObss[-1, 1:]
                #     robot2LowDimLidarCurObs = robot2LidarTimeLowDimRangesVelsObss[-1, 1:]
                #     # convert to 1*12 array
                #     robot1LowDimLidarCurObsRobot2LowDimLidarCurObs = np.concatenate((robot1LowDimLidarCurObs, robot2LowDimLidarCurObs), axis=0)
                #     robot1LowDimLidarCurObsRobot2LowDimLidarCurObs = lidarScalar.transform(
                #         robot1LowDimLidarCurObsRobot2LowDimLidarCurObs.reshape(1, -1))
                #
                #     robot1LowDimLidarPrvObs = robot1LidarTimeLowDimRangesVelsObss[-2, 1:]
                #     robot2LowDimLidarPrvObs = robot2LidarTimeLowDimRangesVelsObss[-2, 1:]
                #     # convert to 1*12 array
                #     robot1LowDimLidarPrvObsRobot2LowDimLidarPrvObs = np.concatenate((robot1LowDimLidarPrvObs, robot2LowDimLidarPrvObs), axis=0)
                #     robot1LowDimLidarPrvObsRobot2LowDimLidarPrvObs = lidarScalar.transform(
                #         robot1LowDimLidarPrvObsRobot2LowDimLidarPrvObs.reshape(1, -1))
                #     robot1LowDimLidarPrvObssRobot2LowDimLidarPrvObss.append(robot1LowDimLidarPrvObsRobot2LowDimLidarPrvObs)
                #
                #     if len(robot1LowDimLidarPrvObssRobot2LowDimLidarPrvObss) > trainingSeqLen:
                #         robot1LowDimLidarPrvObssRobot2LowDimLidarPrvObssLastTrainingSeqLen = np.asarray(
                #             robot1LowDimLidarPrvObssRobot2LowDimLidarPrvObss[-trainingSeqLen:])
                #         # We must give predict an array of sequences, the following line converts prv to an array of such sequences
                #         robot1LowDimLidarPrvObssRobot2LowDimLidarPrvObssLastTrainingSeqLenReshaped = robot1LowDimLidarPrvObssRobot2LowDimLidarPrvObssLastTrainingSeqLen.reshape(
                #             (1, trainingSeqLen, 12))
                #         robot1LowDimLidarPrdObsRobot2LowDimLidarPrdObs = lowDimLidarLstmModel.predict(
                #             robot1LowDimLidarPrvObssRobot2LowDimLidarPrvObssLastTrainingSeqLenReshaped)
                #
                #         robot1LowDimLidarPrdObsRobot2LowDimLidarPrdObs26Reshaped = robot1LowDimLidarPrdObsRobot2LowDimLidarPrdObs.reshape(2, 6)
                #         robot1LowDimLidarCurObsRobot2LowDimLidarCurObs26Reshaped = robot1LowDimLidarCurObsRobot2LowDimLidarCurObs.reshape(2, 6)
                #
                #         lowDimLidarAbnormalityValue = np.linalg.norm(
                #             robot1LowDimLidarPrdObsRobot2LowDimLidarPrdObs26Reshaped - robot1LowDimLidarCurObsRobot2LowDimLidarCurObs26Reshaped)
                #         lowDimLidarTimeAbnormalityValues.append([time, lowDimLidarAbnormalityValue])
                #         if topicRowCounter % configs["plotUpdateRate"] == 0:
                #             plotAll.updateLidarAbnPlot(np.array(lowDimLidarTimeAbnormalityValues))
                prvTime = time
                print(topicRowCounter)
            # PlotPosGpsLidarLive.showPlot(np.array(gpsTimeAbnormalityValues),"GPS")
            # PlotPosGpsLidarLive.showPlot(np.array(lowDimLidarTimeAbnormalityValues),"LIDAR")

    def getLstmTrainedModel(self,sensorName, npInputSeqs=None, npRelevantOutputSeqs=None, clusterLabel=None):
        pathToLstm = "{}/{}-lstm-seq-len-{}-cluster-label-{}.h5".format(testSharedPath,sensorName,trainingSeqLen,clusterLabel)
        if os.path.exists(pathToLstm):
            return load_model(pathToLstm)
        nFeatures = encodersLatentDim

        model = tf.keras.Sequential()

        #This part creates an LSTM layerwith 50 units.
        # The number 50 represents the dimensionality of the output space
        # (i.e., the number of output units or cells in the LSTM layer).
        model.add(LSTM(50, activation='relu', input_shape=(trainingSeqLen, nFeatures)))
        #output matrix
        model.add(Dense(1*nFeatures))

        print(model.layers)

        model.summary()

        model.compile(optimizer=tf.keras.optimizers.Adam(0.01)
                      , loss=tf.keras.losses.MeanSquaredError()
                      , metrics=['accuracy'])
        fittedModel = model.fit(npInputSeqs, npRelevantOutputSeqs, epochs=10, verbose=1)

        plt.plot(fittedModel.history["loss"])
        plt.title("Loss vs. Epoch")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.grid(True)
        plt.show()

        model.save(pathToLstm)
        return model

    def getBestClustersNumElbow(self, npRobotsEncodedObss:np.ndarray):
        startClusterNum = 2
        wcss = []
        for i in range(startClusterNum, 30):
            kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
            kmeans.fit(npRobotsEncodedObss)
            wcss.append(kmeans.inertia_)
        plt.plot(range(startClusterNum, 30), wcss)
        plt.title('Elbow Method')
        plt.xlabel('Number of clusters')
        plt.ylabel('WCSS')
        plt.show()

    def getSilhouetteVisualizer(self, npRobotsEncodedObss:np.ndarray)->int:
        '''
        The value of silhouette score varies from -1 to 1. If the score is 1,
        the cluster is dense and well-separated than other clusters. A value near
        0 represents overlapping clusters with samples very close to the decision
        boundary of the neighbouring clusters. A negative score [-1, 0]
        indicate that the samples might have got assigned to the wrong clusters.

        Parameters
        ----------
        npRobotsEncodedObss

        Returns
        -------

        '''
        startClustersNum = 2
        silScores = []
        for i in range(startClustersNum, 30):
            km = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
            #
            # Fit the KMeans model
            #
            km.fit_predict(npRobotsEncodedObss)
            #
            # Calculate Silhoutte Score
            #
            silScore = silhouette_score(npRobotsEncodedObss, km.labels_, metric='euclidean')
            silScores.append(silScore)
            #
            # Print the score
            #
            print(f'cluster num: {i} Silhouette Score: {silScore}')

        maxClusterNumIndex = silScores.index(max(silScores))
        bestClusterNum = maxClusterNumIndex + startClustersNum
        print(f"Best num of clusters is: {bestClusterNum} with Silhouette value: {silScores[maxClusterNumIndex]}" )

        plt.plot(range(2, 30), silScores)
        plt.title('Silhouette Method')
        plt.xlabel('Number of clusters')
        plt.ylabel('Silhouette score')
        plt.show()

        return bestClusterNum

    def plotClusterRobotEncodedObss(self,labeledNpRobotsEncodedObssClustersDict):
        # plot clusters
        fig = plt.figure()
        ax = Axes3D(fig)

        for clusterLabel in labeledNpRobotsEncodedObssClustersDict:
            npRobotEncodedObss = np.array(labeledNpRobotsEncodedObssClustersDict[clusterLabel])
            print(npRobotEncodedObss.shape)
            ax.scatter(npRobotEncodedObss[:, 0]
                       , npRobotEncodedObss[:, 1]
                       , npRobotEncodedObss[:, 2]
                       , color=TimePosVelObssPlottingUtility.getRandomColor(), marker='.',
                       alpha=0.04,
                       linewidth=1)

        # ax.set_zlim3d(-15, 15)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.show()

    def getClusters(self, npRobotsEncodedObss:np.ndarray):
        # self.getBestClustersNumElbow(npRobotsEncodedObss)
        # self.getSilhouetteVisualizer(npRobotsEncodedObss)

        # Multiply the last three columns of each 2x7 matrix by 20
        npRobotsEncodedObssCopy = np.copy(npRobotsEncodedObss)

        # Use KMeans for clustering without predefining the number of clusters
        kmeans = KMeans(n_clusters=numOfCLusters, random_state=42)
        predictedLabels = kmeans.fit_predict(npRobotsEncodedObssCopy)

        labeledNpRobotsEncodedObssClustersDict = {}
        for prdLabelCounter,curLabel in enumerate (predictedLabels):
            if curLabel not in labeledNpRobotsEncodedObssClustersDict.keys():
                labeledNpRobotsEncodedObssClustersDict[curLabel] = []
            labeledNpRobotsEncodedObssClustersDict[curLabel].append(npRobotsEncodedObss[prdLabelCounter])
        

        self.plotClusterRobotEncodedObss(labeledNpRobotsEncodedObssClustersDict)
        return kmeans,labeledNpRobotsEncodedObssClustersDict


    def getRobotsGpsAutoEncoder(self, npCombinedRobotsTimePosVelsObss:np.ndarray):
        #two robots, 6 val,vel dims
        inputShape = numOfRobots * gpsValVelDim
        obssLen = npCombinedRobotsTimePosVelsObss.shape[0]
        npValVelFlatObss = npCombinedRobotsTimePosVelsObss[:, :, 1:].reshape((obssLen, inputShape))

        scalerBeforeEncoding = StandardScaler()
        npValVelFlatObss = scalerBeforeEncoding.fit_transform(npValVelFlatObss)

        if not os.path.exists(testSharedPath+"gpsEncoder.h5") or not os.path.exists(testSharedPath+"gpsDecoder.h5"):
            encoder = Sequential([
                Dense(8, activation='relu', input_shape=(inputShape ,)),
                Dense(5, activation='relu'),
                Dense(encodersLatentDim, activation='relu')
            ])

            decoder = Sequential([
                Dense(5, activation='relu', input_shape=(encodersLatentDim,)),
                Dense(8, activation='relu'),
                Dense(inputShape , activation=None)
            ])

            autoencoder = Model(inputs=encoder.input, outputs=decoder(encoder.output))
            autoencoder.compile(loss='mse', optimizer='adam')

            print("Fitting the auto encoder ...")
            modelHistory = autoencoder.fit(npValVelFlatObss
                                           , npValVelFlatObss
                                           , epochs=100
                                           , batch_size=32
                                           , verbose=0)
            encoder.save(testSharedPath+"gpsEncoder.h5")
            decoder.save(testSharedPath+"gpsDecoder.h5")
        else:
            encoder = load_model(testSharedPath+"gpsEncoder.h5")
            decoder = load_model(testSharedPath+"gpsDecoder.h5")

        npFlatEncodedObss = encoder.predict(npValVelFlatObss)

        return encoder,decoder,npFlatEncodedObss,scalerBeforeEncoding





if __name__ == "__main__":

    core = Core()

    # arrays of shap (observationsLength*2*7)
    npCombinedRobotsTimeGpsValVelObss, npCombinedRobotsTimeLowDimLidarVelsObss = core.getTrainingSensoryData()
    npCombinedRobotsTimeGpsValVelObss = npCombinedRobotsTimeGpsValVelObss[0:18000]
    gpsEncoder, gpsDecoder, gpsRobotsEncodedObss, gpsScaler = core.getRobotsGpsAutoEncoder(npCombinedRobotsTimeGpsValVelObss)
    gpsClustering, gpsRobotsEncodedObssClustersDict = core.getClusters(gpsRobotsEncodedObss)

    gpsLstmModelsDictByClusteringLabels = {}
    for clusterLabel, npCombinedRobotsGpsObssForACluster in gpsRobotsEncodedObssClustersDict.items():
        gpsInputSeqs, gpsRelevantOutputSeqs = core.getTrainingSequences(npCombinedRobotsGpsObssForACluster)
        gpsLstmModel = core.getLstmTrainedModel("gps", gpsInputSeqs, gpsRelevantOutputSeqs,clusterLabel)
        gpsLstmModelsDictByClusteringLabels[clusterLabel] = gpsLstmModel

    core.loopThroughTestSensoryData(gpsLstmModelsDictByClusteringLabels,gpsClustering,gpsEncoder,gpsScaler)

    # npCombinedRobotsTimeLowDimLidarVelsObss = npCombinedRobotsTimeLowDimLidarVelsObss[0:6000]

    # gpsScalar, gpsInputSeqs, gpsRelevantOutputSeqs = core.getTrainingSequences(npCombinedRobotsTimeGpsVelsObss, trainingSeqLen)
    # gpsLstmModel = core.getLstmTrainedModel("gps", gpsInputSeqs, gpsRelevantOutputSeqs, trainingSeqLen)

    # lstmScalar,lowDimLidarInputSeqs, lowDimLidarRelevantOutputSeqs = core.getTrainingSequences(npCombinedRobotsTimeLowDimLidarVelsObss, trainingSeqLen)
    # lowDimLidarLstmModel = core.getLstmTrainedModel("lidar",lowDimLidarInputSeqs, lowDimLidarRelevantOutputSeqs, trainingSeqLen)

