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
import yaml
from yaml import CLoader
from ctumrs.TimePosVelObssPlottingUtility import TimePosVelObssPlottingUtility
from ctumrs.sensors.liveLocSensorAbn.two.PlotPosGpsLidarLive import PlotPosGpsLidarLive
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
numOfRobots = len(configs["targetRobotIds"])

gpsValVelDim = 6
lidarValVelDim = 1440
encodersLatentDim = 3
trainingSeqLen = 15
numOfCLusters = 15

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

        # Gps settings
        gpsSensorName = "gps_origin"
        gpsTrainingRowsNumLimit = configs["gps_origin"]["trainingRowsNumLimit"]
        gpsGaussianNoiseVar = configs["gps_origin"]["gaussianNoiseVarCo"]

        if not os.path.exists("{}/normalScenarioRobot1TimeGpsValVelObss.pkl".format(testSharedPath)) \
                or not os.path.exists("{}/normalScenarioRobot2TimeGpsValVelObss.pkl".format(testSharedPath)) \
                or not os.path.exists("{}/normalScenarioRobot1TimeLidarValVelObss.pkl".format(testSharedPath)) \
                or not os.path.exists("{}/normalScenarioRobot2TimeLidarValVelObss.pkl".format(testSharedPath)) :
            with open(pathToNormalScenarioYamlFile, "r") as file:
                topicRows = yaml.load_all(file, Loader=CLoader)

                robot1NpTimeLidarValVelObss = []
                robot2NpTimeLidarValVelObss = []
                robot1NpTimeGpsValVelObss = []
                robot2NpTimeGpsValVelObss = []
                targetRobotLidarTopicRowCounter = 0
                targetRobotsGpsTopicRowCounter = 0

                for topicRowCounter, topicRow in enumerate(topicRows):
                    if (targetRobotLidarTopicRowCounter >= lidarTrainingRowsNumLimit) and (targetRobotsGpsTopicRowCounter >= gpsTrainingRowsNumLimit):
                        break
                    robotId, sensorName = Topic.staticGetRobotIdAndSensorName(topicRow)
                    time = Topic.staticGetTimeByTopicDict(topicRow)

                    if targetRobotsGpsTopicRowCounter < gpsTrainingRowsNumLimit:
                        if sensorName == gpsSensorName:
                            gpsX, gpsY, gpsZ = GpsOrigin.staticGetGpsXyz(topicRow)

                            if robotId == targetRobotIds[0]:
                                robot1NpTimeGpsValVelObss.append([time, gpsX, gpsY, gpsZ])
                            elif robotId == targetRobotIds[1]:
                                robot2NpTimeGpsValVelObss.append([time, gpsX, gpsY, gpsZ])

                            targetRobotsGpsTopicRowCounter += 1
                            print("robot: {}, Sensor: {}, count: {}".format(robotId
                                                                            , sensorName,
                                                                            targetRobotsGpsTopicRowCounter))

                    if targetRobotLidarTopicRowCounter < lidarTrainingRowsNumLimit:
                            if sensorName == lidarSensorName:
                                npRanges = RpLidar.staticGetNpRanges(topicRow)

                                if robotId == targetRobotIds[0]:
                                    robot1NpTimeLidarValVelObss.append(np.insert(npRanges, 0, time, axis=0))
                                elif robotId == targetRobotIds[1]:
                                    robot2NpTimeLidarValVelObss.append(np.insert(npRanges, 0, time, axis=0))

                                targetRobotLidarTopicRowCounter += 1
                                print("robot: {}, Sensor: {}, count: {}".format(robotId
                                                                                , sensorName
                                                                                , targetRobotLidarTopicRowCounter))

            robot1NpTimeGpsValVelObss = TimePosRowsDerivativeComputer.computer(robot1NpTimeGpsValVelObss, 1)
            robot2NpTimeGpsValVelObss = TimePosRowsDerivativeComputer.computer(robot2NpTimeGpsValVelObss, 1)

            robot1NpTimeLidarValVelObss = TimePosRowsDerivativeComputer.computer(robot1NpTimeLidarValVelObss, 1)
            robot2NpTimeLidarValVelObss = TimePosRowsDerivativeComputer.computer(robot2NpTimeLidarValVelObss, 1)

            with open('{}/normalScenarioRobot1TimeGpsValVelObss.pkl'.format(testSharedPath), 'wb') as file:
                pickle.dump(robot1NpTimeGpsValVelObss, file)
            with open('{}/normalScenarioRobot2TimeGpsValVelObss.pkl'.format(testSharedPath), 'wb') as file:
                pickle.dump(robot2NpTimeGpsValVelObss, file)
            with open('{}/normalScenarioRobot1TimeLidarValVelObss.pkl'.format(testSharedPath), 'wb') as file:
                pickle.dump(robot1NpTimeLidarValVelObss, file)
            with open('{}/normalScenarioRobot2TimeLidarValVelObss.pkl'.format(testSharedPath), 'wb') as file:
                pickle.dump(robot2NpTimeLidarValVelObss, file)
        else:
            with open('{}/normalScenarioRobot1TimeGpsValVelObss.pkl'.format(testSharedPath), 'rb') as file:
                robot1NpTimeGpsValVelObss =pickle.load(file)
            with open('{}/normalScenarioRobot2TimeGpsValVelObss.pkl'.format(testSharedPath), 'rb') as file:
                robot2NpTimeGpsValVelObss =pickle.load(file)
            with open('{}/normalScenarioRobot1TimeLidarValVelObss.pkl'.format(testSharedPath), 'rb') as file:
                robot1NpTimeLidarValVelObss =pickle.load(file)
            with open('{}/normalScenarioRobot2TimeLidarValVelObss.pkl'.format(testSharedPath), 'rb') as file:
                robot2NpTimeLidarValVelObss = pickle.load(file)


        combinedRobotsTimeGpsVelsObss = []
        for robot1TimeGpsVelsObsCounter,robot1TimeGpsVelsObs in enumerate(robot1NpTimeGpsValVelObss):
            if robot1TimeGpsVelsObsCounter>=len(robot2NpTimeGpsValVelObss):
                break
            robot2TimeGpsValVelObs = robot2NpTimeGpsValVelObss[robot1TimeGpsVelsObsCounter]
            combinedRobotsTimeGpsValVelObs = [robot1TimeGpsVelsObs,robot2TimeGpsValVelObs]
            combinedRobotsTimeGpsVelsObss.append(combinedRobotsTimeGpsValVelObs)
        npCombinedRobotsTimeGpsVelsObss = np.asarray(combinedRobotsTimeGpsVelsObss)


        combinedRobotsTimeLidarValVelObss = []
        for robot1TimeLidarValVelObsCounter, robot1TimeLidarValVelObs in enumerate(robot1NpTimeLidarValVelObss):
            if robot1TimeLidarValVelObsCounter >= len(robot2NpTimeLidarValVelObss):
                break
            robot2TimeLidarValVelObs = robot2NpTimeLidarValVelObss[robot1TimeLidarValVelObsCounter]
            combinedRobotsTimeLidarValVelObs = [robot1TimeLidarValVelObs, robot2TimeLidarValVelObs]
            combinedRobotsTimeLidarValVelObss.append(combinedRobotsTimeLidarValVelObs)
        npCombinedRobotsTimeLidarValVelObss = np.asarray(combinedRobotsTimeLidarValVelObss)

        return npCombinedRobotsTimeGpsVelsObss,npCombinedRobotsTimeLidarValVelObss

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


    def loopThroughTestSensoryData(self, gpsLstmModelsDictByClusteringLabels:dict,gpsClustering:KMeans,gpsEncoder,gpsScaler:StandardScaler
                                   ,lidarLstmModelsDictByClusteringLabels:dict,lidarClustering:KMeans,lidarEncoder,lidarScaler:StandardScaler):
        normalScenarioStartTime = configs["normalScenarioStartTime"]
        # GPS settings
        robot1GpsCounter = 0
        robot2GpsCounter = 0
        gpsTimeAbnormalityValues = []
        robotsGpsEncodedPrvObss = []

        # Lidar settings
        lidarTestScenarioCounterLimit = configs["lidarTestScenarioCounterLimit"]
        robot1LidarCounter = 0
        robot2LidarCounter = 0
        lidarTimeAbnormalityValues = []
        robotsLidarEncodedPrvObss = []

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
                if robot1LidarCounter >= lidarTestScenarioCounterLimit:
                    break

                robotId, sensorName = topicRow["robotId"],topicRow["sensorName"]
                time = topicRow["time"] - normalScenarioStartTime
                if prvTime == 0:
                    prvTime = time

                if prvTime > time:
                    continue

                if sensorName == "gps_origin":
                    npGpsVal = topicRow["npValue"]

                    if robotId == configs["targetRobotIds"][0]:
                        if robot1GpsCounter == 0:
                            robot1TimeGpsValVelObss = np.concatenate(([time], npGpsVal, [0, 0, 0]),axis=0)
                            robot1GpsCounter += 1
                            continue
                        if robot1GpsCounter >= 1:
                            robot1TimeGpsValVelObss = np.vstack(
                                (robot1TimeGpsValVelObss, np.concatenate(([time], npGpsVal, [0, 0, 0]),axis=0)))
                            robot1GpsTimeDiff = robot1TimeGpsValVelObss[-1][0] - robot1TimeGpsValVelObss[-2][0]
                            if robot1GpsTimeDiff == 0:
                                continue
                            robot1GpsDiff = np.subtract(robot1TimeGpsValVelObss[-1][1:int(gpsValVelDim / 2)],
                                                        robot1TimeGpsValVelObss[-2][1:int(gpsValVelDim / 2)])
                            robot1GpsVels = robot1GpsDiff / robot1GpsTimeDiff
                            robot1TimeGpsValVelObss[-1][int(gpsValVelDim / 2) + 1:gpsValVelDim] = robot1GpsVels
                            robot1GpsCounter += 1

                    elif robotId == configs["targetRobotIds"][1]:
                        if robot2GpsCounter == 0:
                            robot2TimeGpsValVelObss = np.concatenate(([time], npGpsVal, np.zeros(int(gpsValVelDim / 2))),axis=0)
                            robot2GpsCounter += 1
                            continue
                        if robot2GpsCounter >= 1:
                            robot2TimeGpsValVelObss = np.vstack(
                                (robot2TimeGpsValVelObss, np.concatenate(([time], npGpsVal, np.zeros(int(gpsValVelDim / 2))),axis=0)))
                            robot2GpsTimeDiff = robot2TimeGpsValVelObss[-1][0] - robot2TimeGpsValVelObss[-2][0]
                            if robot2GpsTimeDiff == 0:
                                continue
                            robot2GpsDiff = np.subtract(robot2TimeGpsValVelObss[-1][1:int(gpsValVelDim / 2)],
                                                        robot2TimeGpsValVelObss[-2][1:int(gpsValVelDim / 2)])
                            robot2GpsVels = robot2GpsDiff / robot2GpsTimeDiff
                            robot2TimeGpsValVelObss[-1][int(gpsValVelDim / 2) + 1:gpsValVelDim] = robot2GpsVels
                            robot2GpsCounter += 1

                    if topicRowCounter % configs["plotUpdateRate"] == 0:
                        plotAll.updateGpsPlot(np.asarray(robot1TimeGpsValVelObss),
                                              np.asarray(robot2TimeGpsValVelObss))

                    if robot1GpsCounter <= 1 or robot2GpsCounter <= 1:
                        continue

                    # gps abn computer
                    robot1GpsCurObs = robot1TimeGpsValVelObss[-1, 1:]
                    robot2GpsCurObs = robot2TimeGpsValVelObss[-1, 1:]
                    #convert to 1*12 array
                    robotsCurGpsFlattedObs = np.array([np.concatenate((robot1GpsCurObs, robot2GpsCurObs), axis=0)])
                    robotsGpsCurFlattedScaledObs = gpsScaler.transform(robotsCurGpsFlattedObs)
                    robotsGpsCurEncodedObs =  gpsEncoder.predict(robotsGpsCurFlattedScaledObs)


                    robot1GpsPrvObs = robot1TimeGpsValVelObss[-2, 1:]
                    robot2GpsPrvObs = robot2TimeGpsValVelObss[-2, 1:]
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
                elif sensorName == "rplidar":
                    npLidarVal = topicRow["npValue"]

                    if robotId == configs["targetRobotIds"][0]:
                        if robot1LidarCounter == 0:
                            robot1TimeLidarValVelObss = np.concatenate(([time], npLidarVal, np.zeros(int(lidarValVelDim / 2))), axis=0)
                            robot1LidarCounter += 1
                            continue
                        if robot1LidarCounter >= 1:
                            robot1TimeLidarValVelObss = np.vstack(
                                (robot1TimeLidarValVelObss, np.concatenate(([time], npLidarVal, np.zeros(int(lidarValVelDim / 2))), axis=0)))
                            robot1LidarTimeDiff = robot1TimeLidarValVelObss[-1][0] - robot1TimeLidarValVelObss[-2][0]
                            if robot1LidarTimeDiff == 0:
                                continue
                            robot1LidarDiff = np.subtract(robot1TimeLidarValVelObss[-1][1:int(lidarValVelDim / 2)],
                                                        robot1TimeLidarValVelObss[-2][1:int(lidarValVelDim / 2)])
                            robot1LidarVels = robot1LidarDiff / robot1LidarTimeDiff
                            # print(robot1LidarVels.shape)
                            # print(robot1TimeLidarValVelObss.shape)
                            robot1TimeLidarValVelObss[-1][int(lidarValVelDim / 2) + 1:lidarValVelDim] = robot1LidarVels
                            robot1LidarCounter += 1

                    elif robotId == configs["targetRobotIds"][1]:
                        if robot2LidarCounter == 0:
                            robot2TimeLidarValVelObss = np.concatenate(([time], npLidarVal, np.zeros(int(lidarValVelDim / 2))), axis=0)
                            robot2LidarCounter += 1
                            continue
                        if robot2LidarCounter >= 1:
                            robot2TimeLidarValVelObss = np.vstack(
                                (robot2TimeLidarValVelObss, np.concatenate(([time], npLidarVal, np.zeros(int(lidarValVelDim / 2))), axis=0)))
                            robot2LidarTimeDiff = robot2TimeLidarValVelObss[-1][0] - robot2TimeLidarValVelObss[-2][0]
                            if robot2LidarTimeDiff == 0:
                                continue
                            robot2LidarDiff = np.subtract(robot2TimeLidarValVelObss[-1][1:int(lidarValVelDim / 2)],
                                                        robot2TimeLidarValVelObss[-2][1:int(lidarValVelDim / 2)])
                            robot2LidarVels = robot2LidarDiff / robot2LidarTimeDiff
                            robot2TimeLidarValVelObss[-1][int(lidarValVelDim / 2) + 1:lidarValVelDim] = robot2LidarVels
                            robot2LidarCounter += 1

                    if robot1LidarCounter <= 1 or robot2LidarCounter <= 1:
                        continue

                    # lidar abn computer
                    robot1LidarCurObs = robot1TimeLidarValVelObss[-1, 1:]
                    robot2LidarCurObs = robot2TimeLidarValVelObss[-1, 1:]
                    # convert to 1*2880 array
                    robotsCurLidarFlattedObs = np.array([np.concatenate((robot1LidarCurObs, robot2LidarCurObs), axis=0)])
                    robotsLidarCurFlattedScaledObs = lidarScaler.transform(robotsCurLidarFlattedObs)
                    robotsLidarCurEncodedObs = lidarEncoder.predict(robotsLidarCurFlattedScaledObs)

                    robot1LidarPrvObs = robot1TimeLidarValVelObss[-2, 1:]
                    robot2LidarPrvObs = robot2TimeLidarValVelObss[-2, 1:]
                    # convert to 1*encoder latent dim  array
                    robotsPrvLidarFlattedObs = np.array([np.concatenate((robot1LidarPrvObs, robot2LidarPrvObs), axis=0)])
                    robotsLidarPrvEncodedObs = lidarScaler.transform(robotsPrvLidarFlattedObs)
                    robotsLidarPrvEncodedObs = lidarEncoder.predict(robotsLidarPrvEncodedObs)
                    robotsLidarEncodedPrvObss.append(robotsLidarPrvEncodedObs)

                    if len(robotsLidarEncodedPrvObss) > trainingSeqLen:
                        # Take the last training seq len
                        robotsLidarEncodedObssLastTrainingSeqLen = np.asarray(robotsLidarEncodedPrvObss[-trainingSeqLen:])
                        # We must give predict an array of sequences, the following line converts prv to an array of such sequences shape (1,15,12) , lstm.predict needs an array of sequences
                        robotsLidarEncodedObssLastTrainingSeqLenReshaped = robotsLidarEncodedObssLastTrainingSeqLen.reshape(
                            (1, trainingSeqLen, encodersLatentDim))
                        bestLidarLstm = self.getRelevantLstmAccordingToClosestCluster(lidarLstmModelsDictByClusteringLabels,
                                                                                    lidarClustering,
                                                                                    robotsLidarEncodedObssLastTrainingSeqLenReshaped[
                                                                                        0])
                        robotsLidarEncodedPrd = bestLidarLstm.predict(robotsLidarEncodedObssLastTrainingSeqLenReshaped)

                        lidarAbnormalityValue = np.linalg.norm(robotsLidarEncodedPrd - robotsLidarCurEncodedObs)
                        lidarTimeAbnormalityValues.append([time, lidarAbnormalityValue])
                        if topicRowCounter % configs["plotUpdateRate"] == 0:
                            plotAll.updateLidarAbnPlot(np.array(lidarTimeAbnormalityValues))
                prvTime = time
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
        plt.title(f"{sensorName} Loss vs. Epoch for LSTM, cluster label: {clusterLabel}")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.grid(True)
        plt.show()

        model.save(pathToLstm)
        return model

    def getBestClustersNumElbow(self, npRobotsEncodedObss:np.ndarray):
        startClusterNum = 2
        finalClusterNum = 20
        wcss = []
        for i in range(startClusterNum, finalClusterNum):
            kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
            kmeans.fit(npRobotsEncodedObss)
            wcss.append(kmeans.inertia_)
        plt.plot(range(startClusterNum, finalClusterNum), wcss)
        plt.title('Elbow Method')
        plt.xlabel('Number of clusters')
        plt.ylabel('WCSS')
        plt.show()

    def getSilhouetteBestClusterNum(self, npRobotsEncodedObss:np.ndarray)->int:
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
        finalClusterNum = 20
        silScores = []
        for i in range(startClustersNum, finalClusterNum):
            km = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
            km.fit_predict(npRobotsEncodedObss)
            silScore = silhouette_score(npRobotsEncodedObss, km.labels_, metric='euclidean')
            silScores.append(silScore)
            #
            # Print the score
            #
            print(f'cluster num: {i} Silhouette Score: {silScore}')

        maxClusterNumIndex = silScores.index(max(silScores))
        bestClusterNum = maxClusterNumIndex + startClustersNum
        print(f"Best num of clusters is: {bestClusterNum} with Silhouette value: {silScores[maxClusterNumIndex]}" )

        plt.plot(range(startClustersNum, finalClusterNum), silScores)
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
        # clusterNum = self.getSilhouetteBestClusterNum(npRobotsEncodedObss)

        # Use KMeans for clustering without predefining the number of clusters
        kmeans = KMeans(n_clusters=numOfCLusters,init='k-means++', max_iter=300, n_init=10, random_state=0)
        predictedLabels = kmeans.fit_predict(npRobotsEncodedObss)

        labeledNpRobotsEncodedObssClustersDict = {}
        for prdLabelCounter,curLabel in enumerate (predictedLabels):
            if curLabel not in labeledNpRobotsEncodedObssClustersDict.keys():
                labeledNpRobotsEncodedObssClustersDict[curLabel] = []
            labeledNpRobotsEncodedObssClustersDict[curLabel].append(npRobotsEncodedObss[prdLabelCounter])
        

        self.plotClusterRobotEncodedObss(labeledNpRobotsEncodedObssClustersDict)
        return kmeans,labeledNpRobotsEncodedObssClustersDict


    def getRobotsAutoEncoder(self, sensorName, npCombinedRobotsTimeValVelObss:np.ndarray):
        #two robots, 6 val,vel dims, remove time dimension
        inputShape = npCombinedRobotsTimeValVelObss.shape[1] * (npCombinedRobotsTimeValVelObss.shape[2]-1)
        npValVelObssLen = npCombinedRobotsTimeValVelObss.shape[0]
        #remove time
        npValVelFlatObss = npCombinedRobotsTimeValVelObss[:, :, 1:].reshape((npValVelObssLen, inputShape))
        print(npValVelFlatObss.shape)
        scalerBeforeEncoding = StandardScaler()
        npValVelFlatObss = scalerBeforeEncoding.fit_transform(npValVelFlatObss)

        if not os.path.exists(testSharedPath+f"{sensorName}Encoder.h5") or not os.path.exists(testSharedPath+f"{sensorName}Decoder.h5"):
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
            encoder.save(testSharedPath+f"{sensorName}Encoder.h5")
            decoder.save(testSharedPath+f"{sensorName}Decoder.h5")
        else:
            encoder = load_model(testSharedPath+f"{sensorName}Encoder.h5")
            decoder = load_model(testSharedPath+f"{sensorName}Decoder.h5")

        npEncodedObss = encoder.predict(npValVelFlatObss)

        return encoder,decoder,npEncodedObss,scalerBeforeEncoding





if __name__ == "__main__":

    core = Core()

    npCombinedRobotsTimeGpsValVelObss, npCombinedRobotsTimeLidarValVelObss = core.getTrainingSensoryData()
    # gps
    npCombinedRobotsTimeGpsValVelObss = npCombinedRobotsTimeGpsValVelObss[0:18000]
    gpsEncoder, gpsDecoder, gpsRobotsEncodedObss, gpsScaler = core.getRobotsAutoEncoder("gps",npCombinedRobotsTimeGpsValVelObss)
    gpsClustering, gpsRobotsEncodedObssClustersDict = core.getClusters(gpsRobotsEncodedObss)

    gpsLstmModelsDictByClusteringLabels = {}
    for clusterLabel, npCombinedRobotsGpsObssForACluster in gpsRobotsEncodedObssClustersDict.items():
        gpsInputSeqs, gpsRelevantOutputSeqs = core.getTrainingSequences(npCombinedRobotsGpsObssForACluster)
        gpsLstmModel = core.getLstmTrainedModel("gps", gpsInputSeqs, gpsRelevantOutputSeqs,clusterLabel)
        gpsLstmModelsDictByClusteringLabels[clusterLabel] = gpsLstmModel

    # LIDAR
    npCombinedRobotsTimeLidarValVelObss = npCombinedRobotsTimeLidarValVelObss[0:6000]
    lidarEncoder, lidarDecoder, lidarRobotsEncodedObss, lidarScaler = core.getRobotsAutoEncoder("lidar",
                                                                                        npCombinedRobotsTimeLidarValVelObss)
    lidarClustering, lidarRobotsEncodedObssClustersDict = core.getClusters(lidarRobotsEncodedObss)

    lidarLstmModelsDictByClusteringLabels = {}
    for clusterLabel, npCombinedRobotsLidarObssForACluster in lidarRobotsEncodedObssClustersDict.items():
        lidarInputSeqs, lidarRelevantOutputSeqs = core.getTrainingSequences(npCombinedRobotsLidarObssForACluster)
        lidarLstmModel = core.getLstmTrainedModel("lidar", lidarInputSeqs, lidarRelevantOutputSeqs, clusterLabel)
        lidarLstmModelsDictByClusteringLabels[clusterLabel] = lidarLstmModel

    #loop
    core.loopThroughTestSensoryData(gpsLstmModelsDictByClusteringLabels,gpsClustering,gpsEncoder,gpsScaler
                                    ,lidarLstmModelsDictByClusteringLabels,lidarClustering,lidarEncoder,lidarScaler
                                    )

