import matplotlib

# test scenario abn compute
import numpy as np
from matplotlib import pyplot as plt
import PyQt5

class PlotAll:

    def __init__(self):
        matplotlib.use("Qt5Agg")
        plt.ion()
        self.__fig, self.__axes = plt.subplots(3, 1)
        self.__lineRobot1Pos, = self.__axes[0].plot(np.random.randn(100), np.random.randn(100))
        self.__lineRobot2Pos, = self.__axes[0].plot(np.random.randn(100), np.random.randn(100))
        self.__axes[0].set_title("Position")


        self.__lineLidar, = self.__axes[1].plot(np.random.randn(100))
        self.__axes[1].set_title("Lidar abnormality")

        self.__lineGps, = self.__axes[2].plot(np.random.randn(100))
        self.__axes[2].set_title("GPS abnormality")

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
        self.__axes[1].relim()
        self.__axes[1].autoscale_view()
        self.__fig.canvas.update()
        self.__fig.canvas.flush_events()
