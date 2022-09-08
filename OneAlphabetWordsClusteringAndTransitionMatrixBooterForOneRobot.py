import numpy as np
from matplotlib import pyplot as plt

from ctumrs.OneAlphabetWordsTransitionMatrix import OneAlphabetWordsTransitionMatrix
from ctumrs.PosVelsClusteringStrgy import PosVelsClusteringStrgy
from ctumrs.TimePosVelObssPlottingUtility import TimePosVelObssPlottingUtility
from ctumrs.TimePosVelObssUtility import TimePosVelObssUtility
from mMath.statistic.Distance import Distance

clustersNum = 75
velocityCoefficient = 10000
obsDim = 6

timePosVelObssUtility = TimePosVelObssPlottingUtility()

basePathToProject = "/home/donkarlo/Dropbox/projs/research/data/self-aware-drones/ctumrs/two-drones/"
scenarioName = "normal-scenario"
sensorName = "gps"
uav1TimePosVelObssFileName = "uav-1-cleaned-gps-pos-vel.txt"

pathToUav1TimePosVelObssFile = basePathToProject + scenarioName +"/" + sensorName +"/" + uav1TimePosVelObssFileName

# Get data for UAV 1
uav1TimePosVelObssDict = TimePosVelObssUtility.getTimePosVelsAndPosVels(pathToUav1TimePosVelObssFile, velocityCoefficient)


uav1PosVelObss = uav1TimePosVelObssDict["timePosVels"][:, 1:obsDim + 1]
# cluster together vectors
posVelObssClusteringStrgy = PosVelsClusteringStrgy(clustersNum, uav1PosVelObss)
fittedClusters = posVelObssClusteringStrgy.getFittedClusters()
# Make the transition matrix
oneAlphabetWordsTransitionMatrix = OneAlphabetWordsTransitionMatrix(posVelObssClusteringStrgy,uav1PosVelObss)
oneAlphabetWordsTransitionMatrix.getNpTransitionMatrix()

# Save the transition matrix

#########loading follow scenarion#########

basePathToProject = "/home/donkarlo/Dropbox/projs/research/data/self-aware-drones/ctumrs/two-drones/"
scenarioName = "follow-scenario"
sensorName = "gps"
uav1TimePosVelObssFileName = "uav-1-cleaned-gps-pos-vel.txt"

pathToUav1TimePosVelObssFile = basePathToProject + scenarioName + "/" + sensorName + "/" + uav1TimePosVelObssFileName

# Get data for UAV 1
uav1TimePosVelObssDict = TimePosVelObssUtility.getTimePosVelsAndPosVels(pathToUav1TimePosVelObssFile, velocityCoefficient)

uav1PosVelObss = uav1TimePosVelObssDict["timePosVels"][:, 1:obsDim + 1]

rangeLimit = 15000
xyCovVal = 2.0e-4
zzCovVal = 4.0e-4
covMtx = np.array([
    [xyCovVal,0,0,0,0,0]
  , [0,xyCovVal,0,0,0,0]
  , [0,0,zzCovVal,0,0,0]
  , [0,0,0,xyCovVal,0,0]
  , [0,0,0,0,xyCovVal,0]
  , [0,0,0,0,0,zzCovVal]
                   ])

#Calculating abnormality values
abnormalityValues = []
for counter,curObs in enumerate(uav1PosVelObss):
    if counter >= 1 and counter <= rangeLimit:
        prvObs = uav1PosVelObss[counter-1]
        prvObsLabel = posVelObssClusteringStrgy.getPredictedLabelByPosVelObs(prvObs)
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


