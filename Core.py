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
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# configs
with open("configs.yaml", "r") as file:
    configs = yaml.load(file, Loader=CLoader)
testSharedPath = "/home/donkarlo/Desktop/lstm/"
numOfRobots = len(configs["targetRobotIds"])

gpsValDim = 3
gpsValVelDim = 2*gpsValDim
lidarValDim = 720
lidarValVelDim = 2*lidarValDim
encodersLatentDim = 2
trainingSeqLen = 15
numOfCLusters = 20
velCo = 20

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

    def getTrainingSequences(self, npRobotsValEncodedVelObss:np.ndarray):
        '''

        Parameters
        ----------
        npRobotsValEncodedVelObss
        trainingSeqsLen

        Returns
        -------
        A np.array of 1*12 matrices as inputSeqs
        '''
        #remove time from all matrices so that we have an array of 2*6 matrices

        inputSeqs = []
        relevantOutputSeqs = []

        for counter in range(len(npRobotsValEncodedVelObss)):
            # get the last index
            lastIndex = counter + trainingSeqLen

            # if lastIndex is greater than length of sequence then break
            if lastIndex > len(npRobotsValEncodedVelObss) - 1:
                break

            # Create input and output sequence
            inputSeq= npRobotsValEncodedVelObss[counter:lastIndex]
            relevantOutputSeq = npRobotsValEncodedVelObss[lastIndex]

            # append seq_X, seq_y in X and y list
            inputSeqs.append(inputSeq)
            relevantOutputSeqs.append(relevantOutputSeq)

        inputSeqs = np.array(inputSeqs)
        relevantOutputSeqs = np.array(relevantOutputSeqs)

        return inputSeqs, relevantOutputSeqs



    def getRelevantLstmAccordingToClosestCluster(self
                                                 , lstmModelsDictByClusteringLabels:dict
                                                 , clustering:KMeans
                                                 , npPrvReshapedPosVelsObssSeqLen:np.ndarray)->LSTM:
        predictionLabels = clustering.predict(npPrvReshapedPosVelsObssSeqLen)
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

    def getEncodedObs(self
                      , npRobot1SensorValObs
                      , npRobot2SensorValObs
                      , sensorEncoder
                      , sensorB4EncodingScaler
                      ):
        robotsFlattedObs = np.array([np.concatenate((npRobot1SensorValObs, npRobot2SensorValObs), axis=0)])
        robotsFlattedScaledObs = sensorB4EncodingScaler.transform(robotsFlattedObs)
        robotsEncodedObs = sensorEncoder.predict(robotsFlattedScaledObs)[0]
        return robotsEncodedObs

    def getAbnVal(self
                  , npRobotsSensorsValEncodedVelObssLastTraningSeqLenPlusOne:np.ndarray
                  , sensorLstmModelsDictByClusteringLabels:dict
                  , sensorClustering:KMeans
                  ):

        if npRobotsSensorsValEncodedVelObssLastTraningSeqLenPlusOne.shape[0] > trainingSeqLen:
            # Take the last training seq len
            npRobotsEncodedObssLastTrainingSeqLen = npRobotsSensorsValEncodedVelObssLastTraningSeqLenPlusOne[-trainingSeqLen:]
            # We must give predict an array of sequences, the following line converts prv to an array of such sequences shape (1,15,2) , lstm.predict needs an array of sequences
            robotsEncodedObssLastTrainingSeqLenReshaped = npRobotsEncodedObssLastTrainingSeqLen.reshape((1, trainingSeqLen, encodersLatentDim*2))
            bestGpsLstm = self.getRelevantLstmAccordingToClosestCluster(sensorLstmModelsDictByClusteringLabels,
                                                                        sensorClustering,
                                                                        robotsEncodedObssLastTrainingSeqLenReshaped[
                                                                            0])
            robotsEncodedPrd = bestGpsLstm.predict(robotsEncodedObssLastTrainingSeqLenReshaped)

            abnVal = np.linalg.norm(robotsEncodedPrd - npRobotsSensorsValEncodedVelObssLastTraningSeqLenPlusOne[-1])
            return abnVal
        else:
            return None


    def loopThroughTestSensoryData(self
                                   ,gpsLstmModelsDictByClusteringLabels:dict
                                   ,gpsClustering:KMeans
                                   ,gpsEncoder
                                   ,gpsScalerB4Encoding:MinMaxScaler
                                   ,gpsEncodedVelCoScaler:MinMaxScaler
                                   , lidarLstmModelsDictByClusteringLabels: dict
                                   , lidarClustering: KMeans
                                   , lidarEncoder
                                   , lidarScalerB4Encoding: MinMaxScaler
                                   , lidarEncodedVelCoScaler: MinMaxScaler
                                   ):
        normalScenarioStartTime = configs["normalScenarioStartTime"]
        # GPS settings
        robot1GpsCounter = 0
        robot2GpsCounter = 0
        gpsTimeAbnVals = []
        robotsGpsTimeValEncodedVelObss = []

        # Lidar settings
        lidarTestScenarioCounterLimit = configs["lidarTestScenarioCounterLimit"]
        robot1LidarCounter = 0
        robot2LidarCounter = 0
        lidarTimeAbnVals = []
        robotsLidarTimeValEncodedVelObss = []

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
                            npRobot1TimeGpsValVelObss = np.concatenate(([time], npValObs, np.zeros(valObsDim)),axis=0)
                            robot1GpsCounter += 1
                            continue
                        elif robot1GpsCounter >= 1:
                            newNpTimeValVelObss = self.getTimeValVelObssFromNewValObs(time, npRobot1TimeGpsValVelObss, npValObs)
                            if newNpTimeValVelObss is not None:
                                npRobot1TimeGpsValVelObss = newNpTimeValVelObss
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
                    #update the GPS navigation - the top one
                    if topicRowCounter % configs["plotUpdateRate"] == 0:
                        plotAll.updateGpsPlot(np.asarray(npRobot1TimeGpsValVelObss),
                                              np.asarray(robot2TimeGpsValVelObss))

                    if robot1GpsCounter <= 1 or robot2GpsCounter <= 1:
                        continue

                    npGpsCurRobotsValEncodedObs = self.getEncodedObs(npRobot1TimeGpsValVelObss[-1, 1:gpsValDim+1]
                                                             ,robot2TimeGpsValVelObss[-1, 1:gpsValDim+1]
                                                             ,gpsEncoder
                                                             ,gpsScalerB4Encoding
                                                             )
                    gpsRobotsMeanTime = (npRobot1TimeGpsValVelObss[-1, 0]+robot2TimeGpsValVelObss[-1, 0])/2
                    npGpsCurRobotsValEncodedObs = np.concatenate(([gpsRobotsMeanTime],npGpsCurRobotsValEncodedObs),axis=0)
                    robotsGpsTimeValEncodedVelObss.append(npGpsCurRobotsValEncodedObs)
                    if(len(robotsGpsTimeValEncodedVelObss)>trainingSeqLen+1):
                        npRobotsGpsTimeValEncodedVelObss = TimePosRowsDerivativeComputer.computer(robotsGpsTimeValEncodedVelObss, velCo)
                        npRobotsGpsValEncodedVelObssScaled = gpsEncodedVelCoScaler.transform(npRobotsGpsTimeValEncodedVelObss[:,1:])
                        # npRobotsGpsValEncodedVelObssScaled = npRobotsGpsTimeValEncodedVelObss[:,1:]
                        # gps abn computer
                        abnVal= self.getAbnVal(npRobotsGpsValEncodedVelObssScaled[-(trainingSeqLen+1):]
                                                ,gpsLstmModelsDictByClusteringLabels
                                                ,gpsClustering)
                        if abnVal is not None:
                            gpsTimeAbnVals.append([time,abnVal])
                            if topicRowCounter % configs["plotUpdateRate"] == 0:
                                plotAll.updateGpsAbnPlot(np.array(gpsTimeAbnVals))

                elif sensorName == "rplidar":
                    if robotId == configs["targetRobotIds"][0]:
                        if robot1LidarCounter == 0:
                            npRobot1TimeLidarValVelObss = np.concatenate(([time], npValObs, np.zeros(valObsDim)),axis=0)
                            robot1LidarCounter += 1
                            continue
                        elif robot1LidarCounter >= 1:
                            newNpTimeValVelObss = self.getTimeValVelObssFromNewValObs(time, npRobot1TimeLidarValVelObss, npValObs)
                            if newNpTimeValVelObss is not None:
                                npRobot1TimeLidarValVelObss = newNpTimeValVelObss
                            else:
                                continue
                            robot1LidarCounter += 1
                    elif robotId == configs["targetRobotIds"][1]:
                        if robot2LidarCounter == 0:
                            robot2TimeLidarValVelObss = np.concatenate(([time], npValObs, np.zeros(valObsDim)),axis=0)
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

                    npLidarCurRobotsValEncodedObs = self.getEncodedObs(npRobot1TimeLidarValVelObss[-1, 1:lidarValDim+1]
                                                             ,robot2TimeLidarValVelObss[-1, 1:lidarValDim+1]
                                                             ,lidarEncoder
                                                             ,lidarScalerB4Encoding
                                                             )
                    lidarRobotsMeanTime = (npRobot1TimeLidarValVelObss[-1, 0]+robot2TimeLidarValVelObss[-1, 0])/2
                    npLidarCurRobotsValEncodedObs = np.concatenate(([lidarRobotsMeanTime],npLidarCurRobotsValEncodedObs),axis=0)
                    robotsLidarTimeValEncodedVelObss.append(npLidarCurRobotsValEncodedObs)
                    if(len(robotsLidarTimeValEncodedVelObss)>trainingSeqLen+1):
                        npRobotsLidarTimeValEncodedVelObss = TimePosRowsDerivativeComputer.computer(robotsLidarTimeValEncodedVelObss, velCo)
                        npRobotsLidarValEncodedVelObssScaled = lidarEncodedVelCoScaler.transform(npRobotsLidarTimeValEncodedVelObss[:,1:])
                        # npRobotsLidarValEncodedVelObssScaled = npRobotsLidarTimeValEncodedVelObss[:,1:]
                        # lidar abn computer
                        abnVal= self.getAbnVal(npRobotsLidarValEncodedVelObssScaled[-(trainingSeqLen+1):]
                                                ,lidarLstmModelsDictByClusteringLabels
                                                ,lidarClustering)
                        if abnVal is not None:
                            lidarTimeAbnVals.append([time,abnVal])
                            if topicRowCounter % configs["plotUpdateRate"] == 0:
                                plotAll.updateLidarAbnPlot(np.array(lidarTimeAbnVals))
                if len(npRobot1TimeGpsValVelObss.shape) == 2:
                    if (npRobot1TimeGpsValVelObss[-1,0]-npRobot1TimeGpsValVelObss[0,0])>10:
                        if np.linalg.norm(npRobot1TimeGpsValVelObss[-1,1:4] - npRobot1TimeGpsValVelObss[0,1:4])<0.1:
                            with open('{}/followScenarioGpsTimeAbnVals.pkl'.format(testSharedPath),
                                      'wb') as file:
                                pickle.dump(gpsTimeAbnVals, file)
                            with open('{}/followScenarioLidarTimeAbnVals.pkl'.format(testSharedPath),
                                      'wb') as file:
                                pickle.dump(lidarTimeAbnVals, file)
                            break

                prvTime = time
            # PlotPosGpsLidarLive.showPlot(np.array(gpsTimeAbnormalityValues),"GPS")
            # PlotPosGpsLidarLive.showPlot(np.array(lowDimLidarTimeAbnormalityValues),"LIDAR")

    def getLstmTrainedModel(self,sensorName, npInputSeqs=None, npRelevantOutputSeqs=None, clusterLabel=None):
        pathToLstm = "{}/{}-lstm-seq-len-{}-cluster-label-{}.h5".format(testSharedPath,sensorName,trainingSeqLen,clusterLabel)
        if os.path.exists(pathToLstm):
            return load_model(pathToLstm)
        nFeatures = encodersLatentDim*2

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
        for clusterLabel in labeledNpRobotsEncodedObssClustersDict:
            npRobotEncodedObss = np.array(labeledNpRobotsEncodedObssClustersDict[clusterLabel])
            plt.scatter(npRobotEncodedObss[:, 0]
                       , npRobotEncodedObss[:, 1]
                       , color=TimePosVelObssPlottingUtility.getRandomColor()
                       , marker='.'
                       , alpha=0.04
                       , linewidth=1)
        plt.xlabel('Latent space dim 1')
        plt.ylabel('Latent space dim 2')
        plt.title(f'Encoded clusters for sensor {sensorName}')
        plt.legend()
        plt.show()

    def getClusters(self, sensorName,npRobotsEncodedObss:np.ndarray):
        # self.getBestClustersNumElbow(npRobotsEncodedObss)
        # clusterNum = self.getSilhouetteBestClusterNum(npRobotsEncodedObss)
        # print(f"for {sensorName} Silhouette cluster num is {clusterNum}")

        # Use KMeans for clustering without predefining the number of clusters
        kmeans = KMeans(n_clusters=numOfCLusters,init='k-means++', max_iter=300, n_init=10, random_state=0)
        predictedLabels = kmeans.fit_predict(npRobotsEncodedObss)

        labeledNpRobotsEncodedObssClustersDict = {}
        for prdLabelCounter,curLabel in enumerate (predictedLabels):
            if curLabel not in labeledNpRobotsEncodedObssClustersDict.keys():
                labeledNpRobotsEncodedObssClustersDict[curLabel] = []
            labeledNpRobotsEncodedObssClustersDict[curLabel].append(npRobotsEncodedObss[prdLabelCounter])

        for label, clusterObss in labeledNpRobotsEncodedObssClustersDict.items():
            print(f"The length of the list for key {label} is: {len(clusterObss)}")

        self.plotClusterRobotEncodedObss(sensorName,labeledNpRobotsEncodedObssClustersDict)
        return kmeans,labeledNpRobotsEncodedObssClustersDict


    def getAutoEncoderLayers(self, robotsNumValVelObsShape:tuple):
        pass

    def getRobotsAutoEncoder(self, sensorName, npCombinedRobotsTimeValObss:np.ndarray):
        #num of robots, val dims, remove time dimension
        robotsNumValObsShape = npCombinedRobotsTimeValObss.shape[1] * (npCombinedRobotsTimeValObss.shape[2] - 1)
        npValObssLen = npCombinedRobotsTimeValObss.shape[0]
        #remove time
        npValFlatObss = npCombinedRobotsTimeValObss[:, :, 1:].reshape((npValObssLen, robotsNumValObsShape))
        # scalerB4Encoding = StandardScaler()
        scalerB4Encoding = MinMaxScaler()
        npValFlatObssScaled = scalerB4Encoding.fit_transform(npValFlatObss)
        # npValFlatObssScaled = npValFlatObss

        if not os.path.exists(testSharedPath+f"{sensorName}Encoder.h5") or not os.path.exists(testSharedPath+f"{sensorName}Decoder.h5"):
            encoder = Sequential([
                Dense(128, activation='relu', input_shape=(robotsNumValObsShape ,)),
                Dense(64, activation='relu'),
                Dense(32, activation='relu'),
                Dense(encodersLatentDim, activation='relu')
            ])

            decoder = Sequential([
                Dense(64, activation='relu', input_shape=(encodersLatentDim,)),
                Dense(128, activation='relu'),
                Dense(256, activation='relu'),
                Dense(robotsNumValObsShape , activation=None)
            ])

            autoencoder = Model(inputs=encoder.input, outputs=decoder(encoder.output))
            autoencoder.compile(loss='mse', optimizer='adam')

            print(f"Fitting the auto encoder for {sensorName} ...")
            modelHistory = autoencoder.fit(npValFlatObssScaled
                                           , npValFlatObssScaled
                                           , epochs=300
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

        npValEncodedObss = encoder.predict(npValFlatObss)

        # Get the average of the first column for all rows in npCombinedRobotsTimeValObss
        timeColumn = np.mean(npCombinedRobotsTimeValObss[:, :, 0],axis=1)
        # Add the average_column as the first column of npEncodedObss
        npTimeValEncodedObss = np.column_stack([timeColumn, npValEncodedObss])

        for i in range(0, 10):
            randIndex = np.random.randint(0, npValFlatObssScaled.shape[0])
            encoderPredicted = encoder.predict(npValFlatObssScaled[randIndex].reshape(1, robotsNumValObsShape))
            reco = decoder.predict(encoderPredicted)[0]
            print("distance:", np.linalg.norm(npValFlatObssScaled[randIndex] - reco))

            # Set the component numbers as x-axis values
            comNum = np.arange(1, npValFlatObssScaled.shape[1]+1)

            # Plot the components for point1 in blue
            plt.plot(comNum, npValFlatObssScaled[randIndex], color='blue', alpha=0.7,marker='o', markersize=10, linestyle='-', label='Actual data')

            # Plot the components for point2 in orange
            plt.plot(comNum, reco, color='orange', alpha=0.7, marker='o', markersize=10, linestyle='-',label='Reconstructed data')

            plt.xlabel('Component Number')
            plt.ylabel('Component Value')
            plt.title('Comparision between actual and reconstructed')
            plt.legend()
            plt.show()

        plt.figure(figsize=(6, 6))
        plt.scatter(npValEncodedObss[:, 0], npValEncodedObss[:, 1], alpha=.8)
        plt.xlabel('Latent Dimension 1')
        plt.ylabel('Latent Dimension 2')
        plt.show()

        return encoder,npTimeValEncodedObss,scalerB4Encoding






if __name__ == "__main__":

    core = Core()

    #Get training data
    npCombinedRobotsTimeGpsValVelObss, npCombinedRobotsTimeLidarValVelObss = core.getTrainingSensoryData()

    # gps
    gpsObssLen = 36000
    npCombinedRobotsTimeGpsValObss = npCombinedRobotsTimeGpsValVelObss[0:gpsObssLen,:,0:gpsValDim+1]
    gpsEncoder, gpsTimeRobotsValEncodedObss, gpsScalerB4Encoding= core.getRobotsAutoEncoder("Gps", npCombinedRobotsTimeGpsValObss)
    npGpsTimeRobotsValEncodedVelObss = TimePosRowsDerivativeComputer.computer(gpsTimeRobotsValEncodedObss, velCo)

    gpsValEncodedVelCoScaler = MinMaxScaler()
    gpsRobotsValEncodedVelObssScaled = gpsValEncodedVelCoScaler.fit_transform(npGpsTimeRobotsValEncodedVelObss[:, 1:])
    # gpsRobotsValEncodedVelObssScaled = npGpsTimeRobotsValEncodedVelObss[:, 1:]


    gpsClustering, gpsRobotsEncodedValVelScaledObssClustersDict = core.getClusters("Gps", gpsRobotsValEncodedVelObssScaled)

    gpsLstmModelsDictByClusteringLabels = {}
    for clusterLabel, gpsValEncodedVelObssForACluster in gpsRobotsEncodedValVelScaledObssClustersDict.items():
        if len(gpsValEncodedVelObssForACluster) > trainingSeqLen:
            gpsInputSeqs, gpsRelevantOutputSeqs = core.getTrainingSequences(gpsValEncodedVelObssForACluster)
            gpsLstmModel = core.getLstmTrainedModel("gps", gpsInputSeqs, gpsRelevantOutputSeqs,clusterLabel)
            gpsLstmModelsDictByClusteringLabels[clusterLabel] = gpsLstmModel
    
    # LIDAR
    lidarObssLen = 36000
    npCombinedRobotsTimeLidarValObss = npCombinedRobotsTimeLidarValVelObss[0:lidarObssLen, :, 0:lidarValDim + 1]
    lidarEncoder, lidarTimeRobotsValEncodedObss,lidarScalerB4Encoding = core.getRobotsAutoEncoder("Lidar", npCombinedRobotsTimeLidarValObss)
    npLidarTimeRobotsValEncodedVelObss = TimePosRowsDerivativeComputer.computer(lidarTimeRobotsValEncodedObss, velCo)

    lidarValEncodedVelCoScaler = MinMaxScaler()
    lidarRobotsValEncodedVelObssScaled = lidarValEncodedVelCoScaler.fit_transform(npLidarTimeRobotsValEncodedVelObss[:, 1:])

    lidarClustering, lidarRobotsEncodedValVelScaledObssClustersDict = core.getClusters("Lidar",
                                                                                   lidarRobotsValEncodedVelObssScaled)

    lidarLstmModelsDictByClusteringLabels = {}
    for clusterLabel, lidarValEncodedVelObssForACluster in lidarRobotsEncodedValVelScaledObssClustersDict.items():
        if len(lidarValEncodedVelObssForACluster) > trainingSeqLen:
            lidarInputSeqs, lidarRelevantOutputSeqs = core.getTrainingSequences(lidarValEncodedVelObssForACluster)
            lidarLstmModel = core.getLstmTrainedModel("lidar", lidarInputSeqs, lidarRelevantOutputSeqs, clusterLabel)
            lidarLstmModelsDictByClusteringLabels[clusterLabel] = lidarLstmModel
    
    #loop
    core.loopThroughTestSensoryData(
        gpsLstmModelsDictByClusteringLabels
        ,gpsClustering
        ,gpsEncoder
        ,gpsScalerB4Encoding
        ,gpsValEncodedVelCoScaler
        ,lidarLstmModelsDictByClusteringLabels
        ,lidarClustering
        ,lidarEncoder
        ,lidarScalerB4Encoding
        ,lidarValEncodedVelCoScaler
    )

