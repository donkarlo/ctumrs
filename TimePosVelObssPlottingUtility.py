import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import random as rand

class TimePosVelObssPlottingUtility():
    @staticmethod
    def getRandomColor():
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
        ax.set_zlim3d(-15, 15)
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
    def plotRobot1And2PosWithLabeledDictClusters(robot1PosVelsLabeledDictClusters: dict
                                                 , robot2PosVelsLabeledDictClusters: dict) -> None:
        fig = plt.figure()
        ax = Axes3D(fig)

        for robot1PosVelLabel in robot1PosVelsLabeledDictClusters:
            robot1PosXs = []
            robot1PosYs = []
            robot1PosZs = []
            for robotPosVel in robot1PosVelsLabeledDictClusters[robot1PosVelLabel]:
                robot1PosXs.append(robotPosVel[0])
                robot1PosYs.append(robotPosVel[1])
                robot1PosZs.append(robotPosVel[2])
            ax.scatter(robot1PosXs
                       , robot1PosYs
                       , robot1PosZs
                       , color=TimePosVelObssPlottingUtility.getRandomColor(), marker='.',
                       alpha=0.04,
                       linewidth=1)

        for robot2PosVelLabel in robot2PosVelsLabeledDictClusters:
            robot2PosXs = []
            robot2PosYs = []
            robot2PosZs = []
            for robotPosVel in robot2PosVelsLabeledDictClusters[robot2PosVelLabel]:
                robot2PosXs.append(robotPosVel[0])
                robot2PosYs.append(robotPosVel[1])
                robot2PosZs.append(robotPosVel[2])
            ax.scatter(robot2PosXs
                       , robot2PosYs
                       , robot2PosZs
                       , color=TimePosVelObssPlottingUtility.getRandomColor(),
                       marker='.', alpha=0.04,
                       linewidth=1)

        # ax.set_zlim3d(-15, 15)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.show()