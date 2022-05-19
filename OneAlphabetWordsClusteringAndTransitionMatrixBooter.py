import numpy as np
from matplotlib import pyplot as plt

from ctumrs.OneAlphabetWordsTransitionMatrix import OneAlphabetWordsTransitionMatrix
from ctumrs.PosVelObssClusteringStrgy import PosVelObssClusteringStrgy
from ctumrs.TimePosVelObssPlottingUtility import TimePosVelObssPlottingUtility
from ctumrs.TimePosVelObssUtility import TimePosVelObssUtility

clustersNum = 75
velocityCoefficient = 10000

timePosVelObssUtility = TimePosVelObssPlottingUtility()

basePathToProject = "/home/donkarlo/Dropbox/projs/research/data/self-aware-drones/ctumrs/two-drones/"
scenarioName = "normal-scenario"
sensorName = "gps"
uav1TimePosVelObssFileName = "uav-1-cleaned-gps-pos-vel.txt"
uav2TimePosVelObssFileName = "uav-2-cleaned-gps-pos-vel.txt"

pathToUav1TimePosVelObssFile = basePathToProject + scenarioName +"/" + sensorName +"/" + uav1TimePosVelObssFileName
pathToUav2TimePosVelObssFile = basePathToProject+scenarioName+"/"+sensorName+"/"+uav2TimePosVelObssFileName

# Get data for UAV 1
uav1TimePosVelObssDict = TimePosVelObssUtility.getTimePosVelsAndPosVels(pathToUav1TimePosVelObssFile, velocityCoefficient)

# Get data for UAV 2
uav2TimePosVelObssDict = TimePosVelObssUtility.getTimePosVelsAndPosVels(pathToUav2TimePosVelObssFile, velocityCoefficient)

# Put related data together in one vector
uav1Uav2PosVelObss=[]
for uav1TimePosVelObsCounter, uav1TimePosVelObs in enumerate(uav1TimePosVelObssDict["timePosVels"]):
    uav2TimePosVelObs = TimePosVelObssUtility.findClosestTimeWiseFollowerTimePosVelToLeaderTimePosVel(uav1TimePosVelObs,uav2TimePosVelObssDict["timePosVels"])
    uav1Uav2PosVelObss.append(np.concatenate((uav1TimePosVelObs[1:4]
                                             , uav2TimePosVelObs[1:4]
                                             , velocityCoefficient*uav1TimePosVelObs[4:7]
                                             , velocityCoefficient*uav2TimePosVelObs[4:7])))



# cluster together vectors
posVelObssClusteringStrgy = PosVelObssClusteringStrgy(clustersNum,uav1Uav2PosVelObss)
fittedClusters = posVelObssClusteringStrgy.getFittedClusters()
# Make the transition matrix
oneAlphabetWordsTransitionMatrix = OneAlphabetWordsTransitionMatrix(posVelObssClusteringStrgy,uav1Uav2PosVelObss)
oneAlphabetWordsTransitionMatrix.getNpTransitionMatrix()

# Save the transition matrix

#########loading follow secanarion#########

basePathToProject = "/home/donkarlo/Dropbox/projs/research/data/self-aware-drones/ctumrs/two-drones/"
scenarioName = "follow-scenario"
sensorName = "gps"
uav1TimePosVelObssFileName = "uav-1-cleaned-gps-pos-vel.txt"
uav2TimePosVelObssFileName = "uav-2-cleaned-gps-pos-vel.txt"

pathToUav1TimePosVelObssFile = basePathToProject + scenarioName +"/" + sensorName +"/" + uav1TimePosVelObssFileName
pathToUav2TimePosVelObssFile = basePathToProject+scenarioName+"/"+sensorName+"/"+uav2TimePosVelObssFileName

# Get data for UAV 1
uav1TimePosVelObssDict = TimePosVelObssUtility.getTimePosVelsAndPosVels(pathToUav1TimePosVelObssFile, velocityCoefficient)

# Get data for UAV 2
uav2TimePosVelObssDict = TimePosVelObssUtility.getTimePosVelsAndPosVels(pathToUav2TimePosVelObssFile, velocityCoefficient)

# Put related data together in one vector
def getGaussianKullbackLieblerDistance(m0, S0, m1, S1):
    # store inv diag covariance of S1 and diff between means
    N = m0.shape[0]
    iS1 = np.linalg.inv(S1)
    diff = m1 - m0

    # kl is made of three terms
    tr_term = np.trace(iS1 @ S0)
    det_term = np.log(np.linalg.det(S1) / np.linalg.det(S0))  # np.sum(np.log(S1)) - np.sum(np.log(S0))
    quad_term = diff.T @ np.linalg.inv(S1) @ diff  # np.sum( (diff*diff) * iS1, axis=1)
    # print(tr_term,det_term,quad_term)
    return .5 * (tr_term + det_term + quad_term - N)

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


uav1Uav2PosVelObss=[]
for uav1TimePosVelObsCounter, uav1TimePosVelObs in enumerate(uav1TimePosVelObssDict["timePosVels"]):
    uav2TimePosVelObs = TimePosVelObssUtility.findClosestTimeWiseFollowerTimePosVelToLeaderTimePosVel(uav1TimePosVelObs,uav2TimePosVelObssDict["timePosVels"])
    uav1Uav2PosVelObss.append(np.concatenate((uav1TimePosVelObs[1:4]
                                             , uav2TimePosVelObs[1:4]
                                             , velocityCoefficient*uav1TimePosVelObs[4:7]
                                             , velocityCoefficient*uav2TimePosVelObs[4:7])))

rangeLimit = 15000
xyCovVal = 2.0e-4
zzCovVal = 4.0e-4
covMtx = np.array([
    [xyCovVal,0,0,0,0,0,0,0,0,0,0,0]
  , [0,xyCovVal,0,0,0,0,0,0,0,0,0,0]
  , [0,0,zzCovVal,0,0,0,0,0,0,0,0,0]
  , [0,0,0,xyCovVal,0,0,0,0,0,0,0,0]
  , [0,0,0,0,xyCovVal,0,0,0,0,0,0,0]
  , [0,0,0,0,0,zzCovVal,0,0,0,0,0,0]
  , [0,0,0,0,0,0,xyCovVal,0,0,0,0,0]
  , [0,0,0,0,0,0,0,xyCovVal,0,0,0,0]
  , [0,0,0,0,0,0,0,0,zzCovVal,0,0,0]
  , [0,0,0,0,0,0,0,0,0,xyCovVal,0,0]
  , [0,0,0,0,0,0,0,0,0,0,xyCovVal,0]
  , [0,0,0,0,0,0,0,0,0,0,0,zzCovVal]
                   ])


abnormalityValues = []
for counter,curObs in enumerate(uav1Uav2PosVelObss):
    if counter >= 1 and counter <= rangeLimit:
        prvObs = uav1Uav2PosVelObss[counter-1]
        prvObsLabel = posVelObssClusteringStrgy.getLabelByPosVelObs(prvObs)
        predictedNextLabel = oneAlphabetWordsTransitionMatrix.getHighestPorobabelNextLabelBasedOnthePrvOne(prvObsLabel)
        predictedNextLabelCenter = posVelObssClusteringStrgy.getClusterCenterByLabel(predictedNextLabel)
        abnormalityValue = getGaussianKullbackLieblerDistance(curObs,covMtx,predictedNextLabelCenter,covMtx)
        print(abnormalityValue)
        abnormalityValues.append(abnormalityValue)

plotAbnormalities(abnormalityValues,rangeLimit)

