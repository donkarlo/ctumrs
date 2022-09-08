import os
import pickle

import numpy as np
from matplotlib import pyplot as plt

from ctumrs.OneAlphabetWordsTransitionMatrix import OneAlphabetWordsTransitionMatrix
from ctumrs.PosVelsClusteringStrgy import PosVelsClusteringStrgy
from ctumrs.sensors.lidar.two.ranges.TimeRangesVelsObss import TimeRangesVelsObss
from mMath.calculus.derivative.TimePosRowsDerivativeComputer import TimePosRowsDerivativeComputer
from mMath.data.preProcess.RowsNormalizer import RowsNormalizer
from mMath.statistic.Distance import Distance
from MachineSettings import MachineSettings
from tensorflow.keras.models import Model, Sequential,load_model
from tensorflow.keras.layers import Dense



rowsNum = 50000
leadership = "follower"
sensorName = "lidar"
scenarioName = "normal"
strgyName = "autoencoders"
clustersNum = 75
dataDim = 3
velCoefficient = 10000
basePathToProject = MachineSettings.MAIN_PATH+"projs/research/data/self-aware-drones/ctumrs/two-drones/"
pathToScenrioSensor = basePathToProject + "{}-scenario/{}/".format(scenarioName,sensorName)


########## Build transition matrix from normal scenario


if not os.path.exists("/home/donkarlo/Desktop/OneRobotFollowerLidarEncodedTranMtx.pkl"):
    # load lidar normal data from relevant robot
    pklFile = open(pathToScenrioSensor + "twoLidarsTimeRangesObss.pkl", "rb")
    robotTimeRangesObss = np.array(pickle.load(pklFile)["followerTimeRangesObss"])
    robotNormalizedRangesObss = RowsNormalizer.getNpNormalizedNpRows(robotTimeRangesObss[:, 1:])
    encoderModel = load_model(filepath=pathToScenrioSensor + "autoencoders/follower-encoder-rows-num-50000-epochs-2160-batch-size-32.h5")
    robotLowDimPosObss = encoderModel.predict(robotNormalizedRangesObss)
    robotTimeRangesVelsObss = np.hstack((robotTimeRangesObss[:, 0:1], robotLowDimPosObss))
    robotLowDimTimePosVelObss = TimePosRowsDerivativeComputer.computer(robotLowDimPosObss, velCoefficient)

    # cluster data
    posVelObssClusteringStrgy = PosVelsClusteringStrgy(clustersNum, robotTimeRangesVelsObss[:, 1:dataDim + 1])
    fittedClusters = posVelObssClusteringStrgy.getFittedClusters()

    # build one alphabet transition matrix
    oneAlphabetWordsTransitionMatrix = OneAlphabetWordsTransitionMatrix(posVelObssClusteringStrgy
                                                                        ,robotTimeRangesVelsObss[:,1:dataDim+1])
    oneAlphabetWordsTransitionMatrix.getNpTransitionMatrix()
    oneAlphabetWordsTransitionMatrix.save("/home/donkarlo/Desktop/OneRobotFollowerLidarEncodedTranMtx.pkl")
else:
    with open("/home/donkarlo/Desktop/OneRobotFollowerLidarEncodedTranMtx.pkl", 'rb') as file:
        oneAlphabetWordsTransitionMatrix = loadedPickle = pickle.load(file)




########## load again normal scenraio leader data
scenarioName = "normal"
pathToScenrioSensor = basePathToProject + "{}-scenario/{}/".format(scenarioName,sensorName)
pklFile = open(pathToScenrioSensor + "twoLidarsTimeRangesObss.pkl", "rb")
robotTimeRangesObss = np.array(pickle.load(pklFile)["followerTimeRangesObss"])
robotNormalizedRangesObss = RowsNormalizer.getNpNormalizedNpRows(robotTimeRangesObss[:, 1:])
encoderModel = load_model(filepath="/home/donkarlo/Dropbox/projs/research/data/self-aware-drones/ctumrs/two-drones/normal-scenario/lidar/autoencoders/leader-encoder-rows-num-50000-epochs-2160-batch-size-32.h5")
robotLowDimPosObss = encoderModel.predict(robotNormalizedRangesObss)
robotTimeRangesVelsObss = np.hstack((robotTimeRangesObss[:, 0:1], robotLowDimPosObss))
robotLowDimTimePosVelObss = TimePosRowsDerivativeComputer.computer(robotLowDimPosObss, velCoefficient)

# detect abnormality
rangeLimit = 15000
abnormalityValues = []
for counter,curObs in enumerate(robotTimeRangesVelsObss[:,1:dataDim+1]):
    if counter >= 1 and counter <= rangeLimit:
        prvObs = robotTimeRangesVelsObss[counter-1][1:dataDim+1]
        prvObsLabel = oneAlphabetWordsTransitionMatrix.getClusteringStrgy().getPredictedLabelByPosVelObs(prvObs)
        predictedNextLabel = oneAlphabetWordsTransitionMatrix.getHighestPorobabelNextLabelBasedOnthePrvOne(prvObsLabel)
        predictedNextLabelCenter = oneAlphabetWordsTransitionMatrix.getClusteringStrgy().getClusterCenterByLabel(predictedNextLabel)
        abnormalityValue = np.linalg.norm(np.array(curObs)-np.array(predictedNextLabelCenter))
        # covCompVal = 0.01
        # covMtx = np.array([[covCompVal,0,0]
        #                   , [0,covCompVal,0]
        #                   , [0,0,covCompVal]]
        #                   )
        # abnormalityValue = Distance.getGaussianKullbackLieblerDistance(np.array(curObs)
        #                                                                ,covMtx
        #                                                                ,np.array(predictedNextLabelCenter)
        #                                                                ,covMtx)
        print(abnormalityValue)
        abnormalityValues.append(abnormalityValue)

def plotAbnormalities(abnormalityValues,rangeLimit):
    # Scale the plot
    f = plt.figure()
    f.set_figwidth(10)
    f.set_figheight(2.5)

    # Label
    plt.xlabel('Timestep')
    plt.ylabel('Abnormality value')
    slicedNoveltyValues = abnormalityValues[0:rangeLimit]
    plt.plot(range(0,rangeLimit)
             , slicedNoveltyValues
             , label=''
             , color='red'
             , linewidth=1)
    # To show xlabel
    plt.tight_layout()

    # To show the inner labels
    plt.legend()

    # Novelty signal
    plt.show()


plotAbnormalities(abnormalityValues,rangeLimit)