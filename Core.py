import pickle

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import silhouette_score
from tensorflow import keras
from tensorflow.keras.models import Model, Sequential,load_model
from tensorflow.keras.layers import Dense,LSTM
from tensorflow.keras import layers, models


import os

from MachineSettings import MachineSettings
import yaml
from yaml import CLoader
from ctumrs.TimePosVelObssPlottingUtility import TimePosVelObssPlottingUtility
from ctumrs.sensors.liveLocSensorAbn.two.PlotPosGpsLidarLive import PlotPosGpsLidarLive
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



    def getRelevantLstmAccordingToClosestCluster(self
                                                 , lstmModelsDictByClusteringLabels:dict
                                                 , clustering:KMeans
                                                 , prvNormalizedReshapedPosVelsObssSeqLen:np.ndarray)->LSTM:
        predictionLabels = clustering.predict(prvNormalizedReshapedPosVelsObssSeqLen)
        mostFrequentLabel = np.bincount(predictionLabels).argmax()
        return lstmModelsDictByClusteringLabels[mostFrequentLabel]


    def getTimeValVelObssFromNewValObs(self, time:int, npTimeValVelObss:np.ndarray, npNewValObs:np.ndarray):
        valObsDim = npNewValObs.shape[0]
        valVelObsDim = 2 * valObsDim
        npNewTimeValVelObss = np.vstack((npTimeValVelObss, np.concatenate(([time], npNewValObs, np.zeros(valObsDim)), axis=0)))
        robot1GpsTimeDiff = npNewTimeValVelObss[-1][0] - npNewTimeValVelObss[-2][0]
        if robot1GpsTimeDiff == 0:
            return None
        robot1GpsDiff = np.subtract(npNewTimeValVelObss[-1][1:valObsDim],
                                    npNewTimeValVelObss[-2][1:valObsDim])
        robot1GpsVels = robot1GpsDiff / robot1GpsTimeDiff
        npNewTimeValVelObss[-1][valObsDim + 1:valVelObsDim] = robot1GpsVels
        return npNewTimeValVelObss

    def getEncodedObs(self, npRobot1SensorValVelObs, npRobot2SensorValVelObs, sensorEncoder, sensorScaler):
        robotsFlattedObs = np.array([np.concatenate((npRobot1SensorValVelObs, npRobot2SensorValVelObs), axis=0)])
        robotsFlattedScaledObs = sensorScaler.transform(robotsFlattedObs)
        robotsEncodedObs = sensorEncoder.predict(robotsFlattedScaledObs)
        return robotsEncodedObs

    def getAbnVal(self
                  , npRobotsSensorsEncodedObssLastTraningSeqLenPlusOne:np.ndarray
                  , sensorLstmModelsDictByClusteringLabels:dict
                  , sensorClustering:KMeans
                  ):

        if len(npRobotsSensorsEncodedObssLastTraningSeqLenPlusOne) > trainingSeqLen:
            # Take the last training seq len
            robotsEncodedObssLastTrainingSeqLen = np.asarray(npRobotsSensorsEncodedObssLastTraningSeqLenPlusOne[-trainingSeqLen:])
            # We must give predict an array of sequences, the following line converts prv to an array of such sequences shape (1,15,12) , lstm.predict needs an array of sequences
            robotsEncodedObssLastTrainingSeqLenReshaped = robotsEncodedObssLastTrainingSeqLen.reshape(
                (1, trainingSeqLen, encodersLatentDim))
            bestGpsLstm = self.getRelevantLstmAccordingToClosestCluster(sensorLstmModelsDictByClusteringLabels,
                                                                        sensorClustering,
                                                                        robotsEncodedObssLastTrainingSeqLenReshaped[
                                                                            0])
            robotsEncodedPrd = bestGpsLstm.predict(robotsEncodedObssLastTrainingSeqLenReshaped)

            abnVal = np.linalg.norm(robotsEncodedPrd - npRobotsSensorsEncodedObssLastTraningSeqLenPlusOne[-1])
            return abnVal
        else:
            return None


    def loopThroughTestSensoryData(self, gpsLstmModelsDictByClusteringLabels:dict,gpsClustering:KMeans,gpsEncoder,gpsScaler:StandardScaler
                                   ,lidarLstmModelsDictByClusteringLabels:dict,lidarClustering:KMeans,lidarEncoder,lidarScaler:StandardScaler):
        normalScenarioStartTime = configs["normalScenarioStartTime"]
        # GPS settings
        robot1GpsCounter = 0
        robot2GpsCounter = 0
        gpsTimeAbnormalityValues = []
        robotsGpsEncodedObss = []

        # Lidar settings
        lidarTestScenarioCounterLimit = configs["lidarTestScenarioCounterLimit"]
        robot1LidarCounter = 0
        robot2LidarCounter = 0
        lidarTimeAbnormalityValues = []
        robotsLidarEncodedObss = []

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
                # How far should I go?
                if robot1LidarCounter >= lidarTestScenarioCounterLimit:
                    break

                robotId, sensorName = topicRow["robotId"],topicRow["sensorName"]
                time = topicRow["time"] - normalScenarioStartTime
                npValObs = topicRow["npValue"]
                valObsDim = npValObs.shape[0]

                if prvTime == 0:
                    prvTime = time

                if prvTime > time:
                    continue

                if sensorName == "gps_origin":
                    if robotId == configs["targetRobotIds"][0]:
                        if robot1GpsCounter == 0:
                            robot1TimeGpsValVelObss = np.concatenate(([time], npValObs, np.zeros(valObsDim)),axis=0)
                            robot1GpsCounter += 1
                            continue
                        elif robot1GpsCounter >= 1:
                            newNpTimeValVelObss = self.getTimeValVelObssFromNewValObs(time, robot1TimeGpsValVelObss, npValObs)
                            if newNpTimeValVelObss is not None:
                                robot1TimeGpsValVelObss = newNpTimeValVelObss
                            else:
                                continue
                            robot1GpsCounter += 1
                    elif robotId == configs["targetRobotIds"][1]:
                        if robot2GpsCounter == 0:
                            robot2TimeGpsValVelObss = np.concatenate(([time], npValObs, np.zeros(valObsDim)),axis=0)
                            robot2GpsCounter += 1
                            continue
                        elif robot2GpsCounter >= 1:
                            newNpTimeValVelObss = self.getTimeValVelObssFromNewValObs(time, robot2TimeGpsValVelObss, npValObs)
                            if newNpTimeValVelObss is not None:
                                robot2TimeGpsValVelObss = newNpTimeValVelObss
                            else:
                                continue
                            robot2GpsCounter += 1

                    if topicRowCounter % configs["plotUpdateRate"] == 0:
                        plotAll.updateGpsPlot(np.asarray(robot1TimeGpsValVelObss),
                                              np.asarray(robot2TimeGpsValVelObss))

                    if robot1GpsCounter <= 1 or robot2GpsCounter <= 1:
                        continue

                    curRobotsEncodedObs = self.getEncodedObs(robot1TimeGpsValVelObss[-1, 1:]
                                                             ,robot2TimeGpsValVelObss[-1, 1:]
                                                             ,gpsEncoder
                                                             ,gpsScaler
                                                             )
                    robotsGpsEncodedObss.append(curRobotsEncodedObs)
                    # gps abn computer
                    abnVal= self.getAbnVal(np.asarray(robotsGpsEncodedObss[-(trainingSeqLen+1):])
                                            ,gpsLstmModelsDictByClusteringLabels
                                            ,gpsClustering)
                    if abnVal is not None:
                        gpsTimeAbnormalityValues.append([time,abnVal])
                        if topicRowCounter % configs["plotUpdateRate"] == 0:
                            plotAll.updateGpsAbnPlot(np.array(gpsTimeAbnormalityValues))

                    # cut length for performance
                    # if robot1TimeGpsValVelObss.shape[0] > 2 * trainingSeqLen:
                    #     robot1TimeGpsValVelObss = robot1TimeGpsValVelObss[-(trainingSeqLen + 2):]
                    #     robot2TimeGpsValVelObss = robot2TimeGpsValVelObss[-(trainingSeqLen + 2):]
                    #     robotsGpsEncodedPrvObss = robotsGpsEncodedPrvObss[-(trainingSeqLen + 2):]
                elif sensorName == "rplidar":
                    npValObs = topicRow["npValue"]

                    if robotId == configs["targetRobotIds"][0]:
                        if robot1LidarCounter == 0:
                            robot1TimeLidarValVelObss = np.concatenate(([time], npValObs, np.zeros(valObsDim)), axis=0)
                            robot1LidarCounter += 1
                            continue
                        elif robot1LidarCounter >= 1:
                            newNpTimeValVelObss = self.getTimeValVelObssFromNewValObs(time, robot1TimeLidarValVelObss, npValObs)
                            if newNpTimeValVelObss is not None:
                                robot1TimeLidarValVelObss = newNpTimeValVelObss
                            else:
                                continue
                            robot1LidarCounter += 1

                    elif robotId == configs["targetRobotIds"][1]:
                        if robot2LidarCounter == 0:
                            robot2TimeLidarValVelObss = np.concatenate(([time], npValObs, np.zeros(valObsDim)), axis=0)
                            robot2LidarCounter += 1
                            continue
                        elif robot2LidarCounter >= 1:
                            newNpTimeValVelObss = self.getTimeValVelObssFromNewValObs(time, robot2TimeLidarValVelObss, npValObs)
                            if newNpTimeValVelObss is not None:
                                robot2TimeLidarValVelObss = newNpTimeValVelObss
                            else:
                                continue
                            robot2LidarCounter += 1

                    if robot1LidarCounter <= 1 or robot2LidarCounter <= 1:
                        continue

                    # lidar abn computer
                    curRobotsEncodedObs = self.getEncodedObs(robot1TimeLidarValVelObss[-1, 1:]
                                                             , robot2TimeLidarValVelObss[-1, 1:]
                                                             , lidarEncoder
                                                             , lidarScaler
                                                             )
                    robotsLidarEncodedObss.append(curRobotsEncodedObs)
                    # lidar abn computer
                    abnVal = self.getAbnVal(np.asarray(robotsLidarEncodedObss[-(trainingSeqLen + 1):])
                                            , lidarLstmModelsDictByClusteringLabels
                                            , lidarClustering)
                    if abnVal is not None:
                        lidarTimeAbnormalityValues.append([time, abnVal])
                        if topicRowCounter % configs["plotUpdateRate"] == 0:
                            plotAll.updateLidarAbnPlot(np.array(lidarTimeAbnormalityValues))

                    # cut length for performance
                    # if robot1TimeLidarValVelObss.shape[0] > 2 * trainingSeqLen:
                    #     robot1TimeLidarValVelObss = robot1TimeLidarValVelObss[-(trainingSeqLen + 2):]
                    #     robot2TimeLidarValVelObss = robot2TimeLidarValVelObss[-(trainingSeqLen + 2):]
                    #     robotsLidarEncodedPrvObss = robotsLidarEncodedPrvObss[-(trainingSeqLen + 2):]
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
        fittedModel = model.fit(npInputSeqs, npRelevantOutputSeqs, epochs=30, verbose=1)

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

    def plotClusterRobotEncodedObss(self,sensorName,labeledNpRobotsEncodedObssClustersDict):
        # plot clusters
        fig = plt.figure()
        ax = Axes3D(fig)

        for clusterLabel in labeledNpRobotsEncodedObssClustersDict:
            npRobotEncodedObss = np.array(labeledNpRobotsEncodedObssClustersDict[clusterLabel])
            ax.scatter(npRobotEncodedObss[:, 0]
                       , npRobotEncodedObss[:, 1]
                       , npRobotEncodedObss[:, 2]
                       , color=TimePosVelObssPlottingUtility.getRandomColor()
                       , marker='.'
                       , alpha=0.04
                       , linewidth=1)
        ax.set_title(f"{sensorName} clusters after encoding.")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.show()

    def getClusters(self, sensorName,npRobotsEncodedObss:np.ndarray):
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
        

        self.plotClusterRobotEncodedObss(sensorName,labeledNpRobotsEncodedObssClustersDict)
        return kmeans,labeledNpRobotsEncodedObssClustersDict


    def getRobotsAutoEncoder(self, sensorName, npCombinedRobotsTimeValVelObss:np.ndarray):
        #num of robots, valvel dims, remove time dimension
        inputShape = npCombinedRobotsTimeValVelObss.shape[1] * (npCombinedRobotsTimeValVelObss.shape[2]-1)
        npValVelObssLen = npCombinedRobotsTimeValVelObss.shape[0]
        #remove time
        npValVelFlatObss = npCombinedRobotsTimeValVelObss[:, :, 1:].reshape((npValVelObssLen, inputShape))
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

            print(f"Fitting the auto encoder for {sensorName} ...")
            modelHistory = autoencoder.fit(npValVelFlatObss
                                           , npValVelFlatObss
                                           , epochs=100
                                           , batch_size=32
                                           , verbose=0)
            encoder.save(testSharedPath+f"{sensorName}Encoder.h5")
            decoder.save(testSharedPath+f"{sensorName}Decoder.h5")

            plt.plot(modelHistory.history["loss"])
            plt.title(f"{sensorName} Loss vs. Epoch for auto encoder")
            plt.ylabel("Loss")
            plt.xlabel("Epoch")
            plt.grid(True)
            plt.show()
        else:
            encoder = load_model(testSharedPath+f"{sensorName}Encoder.h5")
            decoder = load_model(testSharedPath+f"{sensorName}Decoder.h5")

        npEncodedObss = encoder.predict(npValVelFlatObss)

        return encoder,decoder,npEncodedObss,scalerBeforeEncoding

    def getRobotsLidarAutoEncoder(self, sensorName, npCombinedRobotsTimeValVelObss:np.ndarray):
        #num of robots, valvel dims, remove time dimension
        inputShape = npCombinedRobotsTimeValVelObss.shape[1] * (npCombinedRobotsTimeValVelObss.shape[2]-1)
        npValVelObssLen = npCombinedRobotsTimeValVelObss.shape[0]
        #remove time
        npValVelFlatObss = npCombinedRobotsTimeValVelObss[:, :, 1:].reshape((npValVelObssLen, inputShape))
        scalerBeforeEncoding = StandardScaler()
        npValVelFlatObss = scalerBeforeEncoding.fit_transform(npValVelFlatObss)

        if not os.path.exists(testSharedPath+f"{sensorName}Encoder.h5") or not os.path.exists(testSharedPath+f"{sensorName}Decoder.h5"):
            encoder = Sequential([
                Dense(1024, activation='relu', input_shape=(inputShape ,)),
                Dense(512, activation='relu'),
                Dense(256, activation='relu'),
                Dense(128, activation='relu'),
                Dense(64, activation='relu'),
                Dense(32, activation='relu'),
                Dense(encodersLatentDim, activation='relu')
            ])

            decoder = Sequential([
                Dense(32, activation='relu', input_shape=(encodersLatentDim,)),
                Dense(64, activation='relu'),
                Dense(128, activation='relu'),
                Dense(256, activation='relu'),
                Dense(512, activation='relu'),
                Dense(1024, activation='relu'),
                Dense(inputShape , activation=None)
            ])

            autoencoder = Model(inputs=encoder.input, outputs=decoder(encoder.output))
            autoencoder.compile(loss='mse', optimizer='adam')

            print(f"Fitting the auto encoder for {sensorName} ...")
            modelHistory = autoencoder.fit(npValVelFlatObss
                                           , npValVelFlatObss
                                           , epochs=100
                                           , batch_size=32
                                           , verbose=0)
            encoder.save(testSharedPath+f"{sensorName}Encoder.h5")
            decoder.save(testSharedPath+f"{sensorName}Decoder.h5")

            plt.plot(modelHistory.history["loss"])
            plt.title(f"{sensorName} Loss vs. Epoch for auto encoder")
            plt.ylabel("Loss")
            plt.xlabel("Epoch")
            plt.grid(True)
            plt.show()
        else:
            encoder = load_model(testSharedPath+f"{sensorName}Encoder.h5")
            decoder = load_model(testSharedPath+f"{sensorName}Decoder.h5")

        npEncodedObss = encoder.predict(npValVelFlatObss)

        return encoder,decoder,npEncodedObss,scalerBeforeEncoding

    def getRobotsVAEAutoEncoder(self, sensorName, npCombinedRobotsTimeValVelObss:np.ndarray):
        # num of robots, valvel dims, remove time dimension
        inputShape = npCombinedRobotsTimeValVelObss.shape[1] * (npCombinedRobotsTimeValVelObss.shape[2] - 1)
        npValVelObssLen = npCombinedRobotsTimeValVelObss.shape[0]
        # remove time
        npValVelFlatObss = npCombinedRobotsTimeValVelObss[:, :, 1:].reshape((npValVelObssLen, inputShape))
        scalerBeforeEncoding = StandardScaler()
        npValVelFlatObss = scalerBeforeEncoding.fit_transform(npValVelFlatObss)

        # Generate some random data for demonstration purposes
        num_samples = 1000
        input_dim = 720
        data = np.random.rand(num_samples, input_dim)

        # Define the VAE architecture
        latent_dim = 3  # Set the desired latent dimensionality

        # Encoder
        encoder_inputs = layers.Input(shape=(input_dim,))
        x = layers.Dense(256, activation='relu')(encoder_inputs)
        z_mean = layers.Dense(latent_dim, name='z_mean')(x)
        z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)

        # Reparameterization trick to sample from the learned distribution
        def sampling(args):
            z_mean, z_log_var = args
            batch = tf.shape(z_mean)[0]
            dim = tf.shape(z_mean)[1]
            epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
            return z_mean + tf.exp(0.5 * z_log_var) * epsilon

        z = layers.Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

        # Instantiate encoder model
        encoder = models.Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder')

        # Decoder
        decoder_inputs = layers.Input(shape=(latent_dim,))
        x = layers.Dense(256, activation='relu')(decoder_inputs)
        outputs = layers.Dense(input_dim, activation='sigmoid')(x)

        # Instantiate decoder model
        decoder = models.Model(decoder_inputs, outputs, name='decoder')

        # VAE model
        outputs = decoder(encoder(encoder_inputs)[2])
        vae = models.Model(encoder_inputs, outputs, name='vae')

        # Loss function using KL divergence and binary cross-entropy
        kl_loss = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1)
        vae.add_loss(tf.reduce_mean(kl_loss))
        vae.compile(optimizer='adam', loss='binary_crossentropy')

        # Train the VAE on your data
        vae.fit(data, data, epochs=10, batch_size=32)

        # Now, you can use the trained encoder to get mean and log variance for new data
        unseen_input = np.random.rand(1, input_dim)
        mean_val, log_var_val, z_val = encoder.predict(unseen_input)

        print("Mean:", mean_val)
        print("Log Variance:", log_var_val)
        print("Latent Vector:", z_val)





if __name__ == "__main__":

    core = Core()

    npCombinedRobotsTimeGpsValVelObss, npCombinedRobotsTimeLidarValVelObss = core.getTrainingSensoryData()
    # gps
    npCombinedRobotsTimeGpsValVelObss = npCombinedRobotsTimeGpsValVelObss[0:72000]
    gpsEncoder, gpsDecoder, gpsRobotsEncodedObss, gpsScaler = core.getRobotsAutoEncoder("gps", npCombinedRobotsTimeGpsValVelObss)
    gpsClustering, gpsRobotsEncodedObssClustersDict = core.getClusters("gps",gpsRobotsEncodedObss)

    gpsLstmModelsDictByClusteringLabels = {}
    for clusterLabel, npCombinedRobotsGpsObssForACluster in gpsRobotsEncodedObssClustersDict.items():
        gpsInputSeqs, gpsRelevantOutputSeqs = core.getTrainingSequences(npCombinedRobotsGpsObssForACluster)
        gpsLstmModel = core.getLstmTrainedModel("gps", gpsInputSeqs, gpsRelevantOutputSeqs,clusterLabel)
        gpsLstmModelsDictByClusteringLabels[clusterLabel] = gpsLstmModel

    # LIDAR
    npCombinedRobotsTimeLidarValVelObss = npCombinedRobotsTimeLidarValVelObss[0:72000]
    lidarEncoder, lidarDecoder, lidarRobotsEncodedObss, lidarScaler = core.getRobotsLidarAutoEncoder("lidar",
                                                                                                   npCombinedRobotsTimeLidarValVelObss)
    lidarClustering, lidarRobotsEncodedObssClustersDict = core.getClusters("lidar",lidarRobotsEncodedObss)

    lidarLstmModelsDictByClusteringLabels = {}
    for clusterLabel, npCombinedRobotsLidarObssForACluster in lidarRobotsEncodedObssClustersDict.items():
        lidarInputSeqs, lidarRelevantOutputSeqs = core.getTrainingSequences(npCombinedRobotsLidarObssForACluster)
        lidarLstmModel = core.getLstmTrainedModel("lidar", lidarInputSeqs, lidarRelevantOutputSeqs, clusterLabel)
        lidarLstmModelsDictByClusteringLabels[clusterLabel] = lidarLstmModel

    #loop
    core.loopThroughTestSensoryData(gpsLstmModelsDictByClusteringLabels,gpsClustering,gpsEncoder,gpsScaler
                                    ,lidarLstmModelsDictByClusteringLabels,lidarClustering,lidarEncoder,lidarScaler
                                    )

