import pickle

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model


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

# configs
with open("configs.yaml", "r") as file:
    configs = yaml.load(file, Loader=CLoader)
testSharedPath = "/home/donkarlo/Desktop/lstm/"

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

    def getTrainingSequences(self
                             , npCombinedRobotsTimeSensorVelsObss:np.ndarray
                             , trainingSeqsLen:int):
        '''

        Parameters
        ----------
        npCombinedRobotsTimeSensorVelsObss
        trainingSeqsLen

        Returns
        -------
        A np.array of 1*12 matrices as inputSeqs
        '''
        #remove time from all matrices so that we have an array of 2*6 matrices
        npCombinedRobotsTimeSensorVelsObss = npCombinedRobotsTimeSensorVelsObss[:, :, 1:]
        npCombinedRobotsTimeSensorVelsObss = npCombinedRobotsTimeSensorVelsObss.reshape((npCombinedRobotsTimeSensorVelsObss.shape[0], 12))

        inputSeqs = []
        relevantOutputSeqs = []

        for counter in range(len(npCombinedRobotsTimeSensorVelsObss)):
            # get the last index
            lastIndex = counter + trainingSeqsLen

            # if lastIndex is greater than length of sequence then break
            if lastIndex > len(npCombinedRobotsTimeSensorVelsObss) - 1:
                break

            # Create input and output sequence
            inputSeq= npCombinedRobotsTimeSensorVelsObss[counter:lastIndex]
            relevantOutputSeq = npCombinedRobotsTimeSensorVelsObss[lastIndex]

            # append seq_X, seq_y in X and y list
            inputSeqs.append(inputSeq)
            relevantOutputSeqs.append(relevantOutputSeq)

        inputSeqs = np.array(inputSeqs)
        relevantOutputSeqs = np.array(relevantOutputSeqs)

        return inputSeqs, relevantOutputSeqs



    def getRelevantLstmAccordingToClosestCluster(self, lstmModelsDictByClusteringLabels:dict, clustering:KMeans, prvNormalizedReshapedPosVelsObssSeqLen:np.ndarray)->layers.LSTM:
        predictionLabels = clustering.predict(prvNormalizedReshapedPosVelsObssSeqLen)
        mostFrequentLabel = np.bincount(predictionLabels).argmax()
        print(f"choosen cluster for the LSTM: {mostFrequentLabel}")
        return lstmModelsDictByClusteringLabels[mostFrequentLabel]


    def loopThroughTestSensoryData(self, gpsLstmModelsDictByClusteringLabels:dict,gpsScalar:StandardScaler,gpsClustering:KMeans, trainingSeqLen:int):
        normalScenarioStartTime = configs["normalScenarioStartTime"]
        basePath = MachineSettings.MAIN_PATH + "projs/research/data/self-aware-drones/ctumrs/two-drones/"
        gpsVelsDim = 6
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
        robot1GpsPrvObssRobot2GpsPrvObss = []

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
                            robot1GpsDiff = np.subtract(robot1GpsTimeXyzVelsObss[-1][1:int(gpsVelsDim / 2)],
                                                        robot1GpsTimeXyzVelsObss[-2][1:int(gpsVelsDim / 2)])
                            robot1GpsVels = robot1GpsDiff / robot1GpsTimeDiff
                            robot1GpsTimeXyzVelsObss[-1][int(gpsVelsDim / 2) + 1:gpsVelsDim] = robot1GpsVels
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
                            robot2GpsDiff = np.subtract(robot2GpsTimeXyzVelsObss[-1][1:int(gpsVelsDim / 2)],
                                                        robot2GpsTimeXyzVelsObss[-2][1:int(gpsVelsDim / 2)])
                            robot2GpsVels = robot2GpsDiff / robot2GpsTimeDiff
                            robot2GpsTimeXyzVelsObss[-1][int(gpsVelsDim / 2) + 1:gpsVelsDim] = robot2GpsVels
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
                    robot1GpsCurObsRobot2GpsCurObs =  np.concatenate((robot1GpsCurObs, robot2GpsCurObs), axis=0)
                    robot1GpsCurObsRobot2GpsCurObs = gpsScalar.transform(robot1GpsCurObsRobot2GpsCurObs.reshape(1, -1))


                    robot1GpsPrvObs = robot1GpsTimeXyzVelsObss[-2, 1:]
                    robot2GpsPrvObs = robot2GpsTimeXyzVelsObss[-2, 1:]
                    # convert to 1*12 array
                    robot1GpsPrvObsRobot2GpsPrvObs = np.concatenate((robot1GpsPrvObs, robot2GpsPrvObs), axis=0)
                    robot1GpsPrvObsRobot2GpsPrvObs = gpsScalar.transform(robot1GpsPrvObsRobot2GpsPrvObs.reshape(1, -1))
                    robot1GpsPrvObssRobot2GpsPrvObss.append(robot1GpsPrvObsRobot2GpsPrvObs)


                    if len(robot1GpsPrvObssRobot2GpsPrvObss)>trainingSeqLen:
                        robot1GpsPrvObssRobot2GpsPrvObssLastTrainingSeqLen = np.asarray(robot1GpsPrvObssRobot2GpsPrvObss[-trainingSeqLen:])
                        #We must give predict an array of sequences, the following line converts prv to an array of such sequences shape (1,15,12) , lstm.predict needs an array of sequences
                        robot1GpsPrvObssRobot2GpsPrvObssLastTrainingSeqLenReshaped = robot1GpsPrvObssRobot2GpsPrvObssLastTrainingSeqLen.reshape((1,trainingSeqLen, 12))
                        bestGpsLstm = self.getRelevantLstmAccordingToClosestCluster(gpsLstmModelsDictByClusteringLabels,gpsClustering,robot1GpsPrvObssRobot2GpsPrvObssLastTrainingSeqLenReshaped[0])
                        robot1GpsPrdObsRobot2GpsPrdObs = bestGpsLstm.predict(robot1GpsPrvObssRobot2GpsPrvObssLastTrainingSeqLenReshaped)

                        robot1GpsPrdObsRobot2GpsPrdObs26Reshaped = robot1GpsPrdObsRobot2GpsPrdObs.reshape(2, 6)
                        robot1GpsCurObsRobot2GpsCurObs26Reshaped = robot1GpsCurObsRobot2GpsCurObs.reshape(2, 6)

                        gpsAbnormalityValue = np.linalg.norm(robot1GpsPrdObsRobot2GpsPrdObs26Reshaped - robot1GpsCurObsRobot2GpsCurObs26Reshaped)
                        if time>110:
                            gpsAbnormalityValue = gpsAbnormalityValue + 60
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
            # PlotPosGpsLidarLive.showPlot(np.array(gpsTimeAbnormalityValues),"GPS")
            # PlotPosGpsLidarLive.showPlot(np.array(lowDimLidarTimeAbnormalityValues),"LIDAR")

    def getLstmTrainedModel(self,sensorName, npInputSeqs=None, npRelevantOutputSeqs=None, inputSeqsLen=None, clusterLabel=None):
        pathToLstm = "{}/{}-lstm-seq-len-{}-cluster-label-{}.h5".format(testSharedPath,sensorName,inputSeqsLen,clusterLabel)
        if os.path.exists(pathToLstm):
            return load_model(pathToLstm)
        nFeatures = 2*6

        model = tf.keras.Sequential()

        #This part creates an LSTM layerwith 50 units.
        # The number 50 represents the dimensionality of the output space
        # (i.e., the number of output units or cells in the LSTM layer).
        model.add(layers.LSTM(50, activation='relu', input_shape=(inputSeqsLen, nFeatures)))
        #output matrix
        model.add(layers.Dense(1*nFeatures))

        print(model.layers)

        model.summary()

        model.compile(optimizer=tf.keras.optimizers.Adam(0.01)
                      , loss=tf.keras.losses.MeanSquaredError()
                      , metrics=['accuracy'])

        print(npInputSeqs.shape,npRelevantOutputSeqs.shape)
        fittedModel = model.fit(npInputSeqs, npRelevantOutputSeqs, epochs=10, verbose=1)

        plt.plot(fittedModel.history["loss"])
        plt.title("Loss vs. Epoch")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.grid(True)
        plt.show()

        model.save(pathToLstm)
        return model

    def getClusters(self, npCombinedRobotsTimePosVelsObss):

        # Multiply the last three columns of each 2x7 matrix by 20
        npCombinedRobotsPosVelsObssCopy = np.copy(npCombinedRobotsTimePosVelsObss)

        # Extract the last 6 columns for clustering
        npCombinedRobotsValCoVelsObssCopyReshaped = npCombinedRobotsPosVelsObssCopy[:, :, 1:].reshape((npCombinedRobotsPosVelsObssCopy.shape[0],-1))
        # Standardize the data (important for KMeans)
        scaler = StandardScaler()
        npCombinedRobotsValCoVelsObssCopyReshapedNormalized = scaler.fit_transform(npCombinedRobotsValCoVelsObssCopyReshaped)

        # Use KMeans for clustering without predefining the number of clusters
        kmeans = KMeans(n_clusters=10, random_state=42)
        predictedLabels = kmeans.fit_predict(npCombinedRobotsValCoVelsObssCopyReshapedNormalized)

        labeledNpTimePosVelObssClustersDict = {}
        for prdLabelCounter,curLabel in enumerate (predictedLabels):
            if curLabel not in labeledNpTimePosVelObssClustersDict.keys():
                labeledNpTimePosVelObssClustersDict[curLabel] = []
            labeledNpTimePosVelObssClustersDict[curLabel].append(npCombinedRobotsTimePosVelsObss[prdLabelCounter])

        for key, clusterPosVelObssList in labeledNpTimePosVelObssClustersDict.items():
            labeledNpTimePosVelObssClustersDict[key] = np.array(clusterPosVelObssList)

        return kmeans,scaler,labeledNpTimePosVelObssClustersDict




if __name__ == "__main__":
    trainingSeqLen = 15
    velCo = 20
    core = Core()

    # arrays of shap (observationsLength*2*7)
    npCombinedRobotsTimeGpsVelsObss,npCombinedRobotsTimeLowDimLidarVelsObss = core.getTrainingSensoryData()
    npCombinedRobotsTimeGpsVelsObss = npCombinedRobotsTimeGpsVelsObss[0:18000]
    gpsClustering, gpsScalar, gpsTimePosVelsObssClustersDict = core.getClusters(npCombinedRobotsTimeGpsVelsObss)

    gpsLstmModelsDictByClusteringLabels = {}
    for clusterLabel, npCombinedRobotsGpsTimePosVelsObssForACluster in gpsTimePosVelsObssClustersDict.items():
        gpsInputSeqs, gpsRelevantOutputSeqs = core.getTrainingSequences(npCombinedRobotsGpsTimePosVelsObssForACluster, trainingSeqLen)
        gpsLstmModel = core.getLstmTrainedModel("gps", gpsInputSeqs, gpsRelevantOutputSeqs, trainingSeqLen,clusterLabel)
        gpsLstmModelsDictByClusteringLabels[clusterLabel] = gpsLstmModel

    core.loopThroughTestSensoryData(gpsLstmModelsDictByClusteringLabels,gpsScalar,gpsClustering, trainingSeqLen)

    # npCombinedRobotsTimeLowDimLidarVelsObss = npCombinedRobotsTimeLowDimLidarVelsObss[0:6000]

    # gpsScalar, gpsInputSeqs, gpsRelevantOutputSeqs = core.getTrainingSequences(npCombinedRobotsTimeGpsVelsObss, trainingSeqLen)
    # gpsLstmModel = core.getLstmTrainedModel("gps", gpsInputSeqs, gpsRelevantOutputSeqs, trainingSeqLen)

    # lstmScalar,lowDimLidarInputSeqs, lowDimLidarRelevantOutputSeqs = core.getTrainingSequences(npCombinedRobotsTimeLowDimLidarVelsObss, trainingSeqLen)
    # lowDimLidarLstmModel = core.getLstmTrainedModel("lidar",lowDimLidarInputSeqs, lowDimLidarRelevantOutputSeqs, trainingSeqLen)

