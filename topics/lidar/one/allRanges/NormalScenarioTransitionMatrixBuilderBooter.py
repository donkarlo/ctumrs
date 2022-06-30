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
# load lidar normal data from relevant robot
pklFile = open(pathToScenrioSensor + "twoLidarsTimeRangesVelsObss.pkl", "rb")
robotTimeRangesVelsObss = pickle.load(pklFile)["leaderTimeRangesVelsObss"]
robotTimeRangesVelsObss = TimeRangesVelsObss.velMulInTimeRangesVelsObss(np.array(robotTimeRangesVelsObss)
                                                                        ,velMulCo)
# cluster data
posVelObssClusteringStrgy = TimePosVelsClusteringStrgy(clustersNum, robotTimeRangesVelsObss[:,1:dataDim+1])
fittedClusters = posVelObssClusteringStrgy.getFittedClusters()

# build one alphabet transition matrix
oneAlphabetWordsTransitionMatrix = OneAlphabetWordsTransitionMatrix(posVelObssClusteringStrgy
                                                                    ,robotTimeRangesVelsObss[:,1:dataDim+1])
oneAlphabetWordsTransitionMatrix.getNpTransitionMatrix()
oneAlphabetWordsTransitionMatrix.save("/home/donkarlo/Desktop/OneRobotLidarTranMtx.txt")


########## load again normal scenraio leader data
secnarionName = "normal"
pathToScenrioSensor = basePathToProject + "{}-scenario/{}/{}/".format(scenarioName,sensorName,strgyName)
robotTimeRangesVelsObss = pickle.load(pklFile)["leaderTimeRangesVelsObss"]
robotTimeRangesVelsObss = TimeRangesVelsObss.velMulInTimeRangesVelsObss(np.array(robotTimeRangesVelsObss)
                                                                        ,velMulCo)
# detect abnormality
rangeLimit = 15000
covVal = 0.01
covMtx = covVal*np.identity(len(robotTimeRangesVelsObss[:,1:dataDim+1]))
abnormalityValues = []
for counter,curObs in enumerate(robotTimeRangesVelsObss):
    if counter >= 1 and counter <= rangeLimit:
        prvObs = robotTimeRangesVelsObss[counter-1]
        prvObsLabel = posVelObssClusteringStrgy.getLabelByPosVelObs(prvObs)
        predictedNextLabel = oneAlphabetWordsTransitionMatrix.getHighestPorobabelNextLabelBasedOnthePrvOne(prvObsLabel)
        predictedNextLabelCenter = posVelObssClusteringStrgy.getClusterCenterByLabel(predictedNextLabel)
        abnormalityValue = Distance.getGaussianKullbackLieblerDistance(curObs,covMtx,predictedNextLabelCenter,covMtx)
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
