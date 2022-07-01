import os
import pickle

import numpy as np
from matplotlib import pyplot as plt

from ctumrs.OneAlphabetWordsTransitionMatrix import OneAlphabetWordsTransitionMatrix
from ctumrs.TimePosVelsClusteringStrgy import TimePosVelsClusteringStrgy
from ctumrs.topics.lidar.two.ranges.TimeRangesVelsObss import TimeRangesVelsObss
from mMath.statistic.Distance import Distance
from MachineSettings import MachineSettings



leadership = "leader"
sensorName = "lidar"
scenarioName = "normal"
strgyName = "ranges"
velMulCo = 10000
clustersNum = 75
dataDim = 1440
basePathToProject = format(MachineSettings.MAIN_PATH)+"projs/research/data/self-aware-drones/ctumrs/two-drones/"
pathToScenrioSensor = basePathToProject + "{}-scenario/{}/{}/".format(scenarioName,sensorName,strgyName)



########## Build transition matrix from normal scenario


if not os.path.exists("/home/donkarlo/Desktop/OneRobotLeaderLidarTranMtx.pkl"):
    # load lidar normal data from relevant robot
    pklFile = open(pathToScenrioSensor + "twoLidarsTimeRangesVelsObss.pkl", "rb")
    robotTimeRangesVelsObss = pickle.load(pklFile)["leaderTimeRangesVelsObss"]
    robotTimeRangesVelsObss = TimeRangesVelsObss.velMulInTimeRangesVelsObss(np.array(robotTimeRangesVelsObss)
                                                                            , velMulCo)

    # cluster data
    posVelObssClusteringStrgy = TimePosVelsClusteringStrgy(clustersNum, robotTimeRangesVelsObss[:, 1:dataDim + 1])
    fittedClusters = posVelObssClusteringStrgy.getFittedClusters()

    # build one alphabet transition matrix
    oneAlphabetWordsTransitionMatrix = OneAlphabetWordsTransitionMatrix(posVelObssClusteringStrgy
                                                                        ,robotTimeRangesVelsObss[:,1:dataDim+1])
    oneAlphabetWordsTransitionMatrix.getNpTransitionMatrix()
    oneAlphabetWordsTransitionMatrix.save("/home/donkarlo/Desktop/OneRobotLeaderLidarTranMtx.pkl")
else:
    with open("/home/donkarlo/Desktop/OneRobotLeaderLidarTranMtx.pkl", 'rb') as file:
        oneAlphabetWordsTransitionMatrix = loadedPickle = pickle.load(file)




########## load again normal scenraio leader data
scenarioName = "follow"
pathToScenrioSensor = basePathToProject + "{}-scenario/{}/{}/".format(scenarioName,sensorName,strgyName)
pklFile = open(pathToScenrioSensor + "twoLidarsTimeRangesVelsObss.pkl", "rb")
robotTimeRangesVelsObss = pickle.load(pklFile)["leaderTimeRangesVelsObss"]
robotTimeRangesVelsObss = TimeRangesVelsObss.velMulInTimeRangesVelsObss(np.array(robotTimeRangesVelsObss)
                                                                        ,velMulCo)
# detect abnormality
rangeLimit = 15000
abnormalityValues = []
for counter,curObs in enumerate(robotTimeRangesVelsObss[:,1:dataDim+1]):
    if counter >= 1 and counter <= rangeLimit:
        prvObs = robotTimeRangesVelsObss[counter-1][1:dataDim+1]
        prvObsLabel = oneAlphabetWordsTransitionMatrix.getClusteringStrgy().getLabelByPosVelObs(prvObs)
        predictedNextLabel = oneAlphabetWordsTransitionMatrix.getHighestPorobabelNextLabelBasedOnthePrvOne(prvObsLabel)
        predictedNextLabelCenter = oneAlphabetWordsTransitionMatrix.getClusteringStrgy().getClusterCenterByLabel(predictedNextLabel)
        abnormalityValue = np.linalg.norm(np.array(curObs)-np.array(predictedNextLabelCenter))
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
