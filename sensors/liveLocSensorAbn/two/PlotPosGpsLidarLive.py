import matplotlib

import numpy as np
from matplotlib import pyplot as plt
import PyQt5

class PlotPosGpsLidarLive:

    def __init__(self):
        tickNumbersFontSize = 12
        matplotlib.use("Qt5Agg")
        plt.ion()
        self.__fig, self.__axes = plt.subplots(3, 1)
        self.__lineRobot1Pos, = self.__axes[0].plot(np.random.randn(100), np.random.randn(100))
        self.__lineRobot2Pos, = self.__axes[0].plot(np.random.randn(100), np.random.randn(100))
        # self.__axes[0].grid(True)
        self.__axes[0].tick_params(axis='both', labelsize=tickNumbersFontSize)
        self.__axes[0].set_title("Position")


        self.__lineLidar, = self.__axes[1].plot(np.random.randn(100))
        # self.__lineTolLidar, = self.__axes[1].plot([0,0],[0,0])
        # self.__axes[1].grid(True)
        self.__axes[1].tick_params(axis='both', labelsize=tickNumbersFontSize)
        self.__axes[1].set_title("LIDAR anomalies")

        self.__lineGps, = self.__axes[2].plot(np.random.randn(100))
        # self.__axes[2].grid(True)
        self.__axes[2].tick_params(axis='both', labelsize=tickNumbersFontSize)
        self.__axes[2].set_title("GPS anomalies")

        plt.show(block=False)

    def updateGpsPlot(self, robot1GpsTimeRows,robot2GpsTimeRows):
        self.__lineRobot1Pos.set_xdata(robot1GpsTimeRows[:, 1])
        self.__lineRobot1Pos.set_ydata(robot1GpsTimeRows[:, 2])

        self.__lineRobot2Pos.set_xdata(robot2GpsTimeRows[:, 1])
        self.__lineRobot2Pos.set_ydata(robot2GpsTimeRows[:, 2])

        self.__axes[0].relim()
        self.__axes[0].autoscale_view()
        self.__fig.canvas.update()
        self.__fig.canvas.flush_events()

    def updateGpsAbnPlot(self, gpsTimeAbnVals):
        self.__lineGps.set_xdata(gpsTimeAbnVals[:, 0])
        self.__lineGps.set_ydata(gpsTimeAbnVals[:, 1])
        self.__axes[2].relim()
        self.__axes[2].autoscale_view()
        self.__fig.canvas.update()
        self.__fig.canvas.flush_events()

    def updateLidarAbnPlot(self, lidarTimeAbnVals):
        self.__lineLidar.set_xdata(lidarTimeAbnVals[:, 0])
        self.__lineLidar.set_ydata(lidarTimeAbnVals[:, 1])

        # self.__lineTolLidar.set_xdata([0,len(lidarTimeAbnVals[:, 0])])
        # self.__lineTolLidar.set_ydata([20,20])

        self.__axes[1].relim()
        self.__axes[1].autoscale_view()
        self.__fig.canvas.update()
        self.__fig.canvas.flush_events()

    @staticmethod
    def showPlot(timAbnVals,sensorName):
        matplotlib.use("Qt5Agg")
        time = timAbnVals[:, 0]
        abnVal = timAbnVals[:, 1]
        plt.plot(time,abnVal)
        plt.xlabel("time")
        plt.ylabel(sensorName)
        plt.grid(True)
        plt.show()
