import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import random as rand


def getRandomColor():
    return [rand.uniform(0, 1.0) for i in [1, 2, 3]]

def plotPos(uavPosVels:list):
    uavPosXs = []
    uavPosYs = []
    uavPosZs = []

    for uavPosVel in uavPosVels:
        uavPosXs.append(uavPosVel[0])
        uavPosYs.append(uavPosVel[1])
        uavPosZs.append(uavPosVel[2])
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(uavPosXs, uavPosYs, uavPosZs, marker='.', alpha=0.04, linewidth=1)
    plt.show()

def plotPosWithCLusters(uavTimePosVelClusters:dict)->None:
    fig = plt.figure()
    ax = Axes3D(fig)
    for uavPosVelLabel in uavTimePosVelClusters:
        uavTimePosXs = []
        uavTimePosYs = []
        uavTimePosZs = []
        for uavTimePosVel in uavTimePosVelClusters[uavPosVelLabel]:
            uavTimePosXs.append(uavTimePosVel[1])
            uavTimePosYs.append(uavTimePosVel[2])
            uavTimePosZs.append(uavTimePosVel[3])
        ax.scatter(uavTimePosXs, uavTimePosYs, uavTimePosZs, color=getRandomColor(), marker='.', alpha=0.04,
                   linewidth=1)
    plt.show()

def plotLeaderFollowerUavPosWithCLusters(leaderUavTimePosVelClusters: dict,
                                         followerUavTimePosVelClusters: dict) -> None:
    fig = plt.figure()
    ax = Axes3D(fig)

    for leaderUavPosVelLabel in leaderUavTimePosVelClusters:
        leaderUavTimePosXs = []
        leaderUavTimePosYs = []
        leaderUavTimePosZs = []
        for uavTimePosVel in leaderUavTimePosVelClusters[leaderUavPosVelLabel]:
            leaderUavTimePosXs.append(uavTimePosVel[1])
            leaderUavTimePosYs.append(uavTimePosVel[2])
            leaderUavTimePosZs.append(uavTimePosVel[3])
        ax.scatter(leaderUavTimePosXs, leaderUavTimePosYs, leaderUavTimePosZs, color=getRandomColor(), marker='.',
                   alpha=0.04,
                   linewidth=1)

    for followerUavPosVelLabel in followerUavTimePosVelClusters:
        followerUavTimePosXs = []
        followerUavTimePosYs = []
        followerUavTimePosZs = []
        for uavTimePosVel in followerUavTimePosVelClusters[followerUavPosVelLabel]:
            followerUavTimePosXs.append(uavTimePosVel[1])
            followerUavTimePosYs.append(uavTimePosVel[2])
            followerUavTimePosZs.append(uavTimePosVel[3])
        ax.scatter(followerUavTimePosXs, followerUavTimePosYs, followerUavTimePosZs, color=getRandomColor(),
                   marker='.', alpha=0.04,
                   linewidth=1)

    ax.set_zlim3d(-15, 15)
    plt.show()

def getTimePosVelsAndPosVels(pathToUavTimePosVelDataFile:str, velocityCoefficient: float)-> None:
    uavTimePosVelDataFile = open(pathToUavTimePosVelDataFile, 'r')
    uavTimePosVelDataFileLines = uavTimePosVelDataFile.readlines()

    # Strips the newline character
    uavPosVels = []
    uavTimePosVels = []
    for uavTimePosVelDataFileLine in uavTimePosVelDataFileLines:
        uavStrTime, \
        uavStrPosX, \
        uavStrPosY, \
        uavStrPosZ, \
        uavStrVelX, \
        uavStrVelY, \
        uavStrVelZ = uavTimePosVelDataFileLine.strip().split(",")

        uavTime = float(uavStrTime)

        uavPosX = float(uavStrPosX)
        uavPosY = float(uavStrPosY)
        uavPosZ = float(uavStrPosZ)

        uavVelX = velocityCoefficient * float(uavStrVelX)
        uavVelY = velocityCoefficient * float(uavStrVelY)
        uavVelZ = velocityCoefficient * float(uavStrVelZ)

        uavPosVels.append([uavPosX
                               , uavPosY
                               , uavPosZ
                               , uavVelX
                               , uavVelY
                               , uavVelZ])

        uavTimePosVels.append([uavTime
                                   , uavPosX
                                   , uavPosY
                                   , uavPosZ
                                   , uavVelX
                                   , uavVelY
                                   , uavVelZ])

    uavPosVels = np.array(uavPosVels)
    uavTimePosVels = np.array(uavTimePosVels)
    return {'posVels':uavPosVels,'timePosVels':uavTimePosVels}


def getPosVelByTimePosVel(timePosVel:list)->np.array:
    return [timePosVel[1],timePosVel[2],timePosVel[3],timePosVel[4],timePosVel[5],timePosVel[6]]


def findClosestTimeWiseFollowerTimePosVelToLeaderTimePosVel(leaderTimePosVel, followerTimePosVels)->int:
    start = 0
    end = len(followerTimePosVels)-1

    while end-start>=3:
        mid = start+int((end-start)/2)
        if leaderTimePosVel[0] == followerTimePosVels[mid][0]:
            return followerTimePosVels[mid]
        elif leaderTimePosVel[0]>followerTimePosVels[mid][0]:
            start = mid
            end = end
        elif leaderTimePosVel[0]<followerTimePosVels[mid][0]:
            start = start
            end = mid
    return followerTimePosVels[end-1]