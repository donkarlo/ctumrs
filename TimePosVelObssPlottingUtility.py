import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import random as rand

class TimePosVelObssPlottingUtility():
    @staticmethod
    def getRandomColor(self):
        return [rand.uniform(0, 1.0) for i in [1, 2, 3]]

    @staticmethod
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

    @staticmethod
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
            ax.scatter(uavTimePosXs, uavTimePosYs, uavTimePosZs, color=TimePosVelObssPlottingUtility.getRandomColor(), marker='.', alpha=0.04,
                       linewidth=1)
        plt.show()

    @staticmethod
    def plotLeaderFollowerUavPosWithCLusters(leaderUavTimePosVelClusters: dict
                                             , followerUavTimePosVelClusters: dict) -> None:
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
            ax.scatter(leaderUavTimePosXs, leaderUavTimePosYs, leaderUavTimePosZs, color=TimePosVelObssPlottingUtility.getRandomColor(), marker='.',
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
            ax.scatter(followerUavTimePosXs, followerUavTimePosYs, followerUavTimePosZs, color=self.getRandomColor(),
                       marker='.', alpha=0.04,
                       linewidth=1)

        ax.set_zlim3d(4.9, 5.1)
        plt.show()