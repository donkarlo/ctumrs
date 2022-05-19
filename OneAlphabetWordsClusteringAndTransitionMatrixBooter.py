import numpy as np

from ctumrs.OneAlphabetWordsTransitionMatrix import OneAlphabetWordsTransitionMatrix
from ctumrs.PosVelObssClusteringStrgy import PosVelObssClusteringStrgy
from ctumrs.TimePosVelObssPlottingUtility import TimePosVelObssPlottingUtility
from ctumrs.TimePosVelObssUtility import TimePosVelObssUtility

clustersNum = 75
velocityCoefficient = 10000

utility = TimePosVelObssPlottingUtility()

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