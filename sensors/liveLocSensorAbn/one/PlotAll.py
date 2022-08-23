import matplotlib

# test scenario abn compute
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import PyQt5


class PlotAll:

    def __init__(self):
        matplotlib.use("Qt5Agg")
        plt.ion()
        self.__fig, self.__axes = plt.subplots(3, 1)
        self.__line0, = self.__axes[0].plot(np.random.randn(100),np.random.randn(100))
        self.__line1, = self.__axes[1].plot(np.random.randn(100))
        self.__line2, = self.__axes[2].plot(np.random.randn(100))
        plt.show(block=False)

    def updateGpsPlot(self,robotSpecificGpsTimeRows):
        self.__line0.set_xdata(robotSpecificGpsTimeRows[:, 1])
        self.__line0.set_ydata(robotSpecificGpsTimeRows[:, 2])
        self.__axes[0].relim()
        self.__axes[0].autoscale_view()
        self.__fig.canvas.update()
        self.__fig.canvas.flush_events()

    def updateLidarAbnPlot(self, lidarTimeAbnVals):
        #Order by time
        self.__line1.set_xdata(lidarTimeAbnVals[:, 0])
        self.__line1.set_ydata(lidarTimeAbnVals[:, 1])
        self.__axes[1].relim()
        self.__axes[1].autoscale_view()
        self.__fig.canvas.update()
        self.__fig.canvas.flush_events()


    def updateGpsAbnPlot(self, lidarTimeAbnVals):
        #Order by time
        self.__line2.set_xdata(lidarTimeAbnVals[:, 0])
        self.__line2.set_ydata(lidarTimeAbnVals[:, 1])
        self.__axes[2].relim()
        self.__axes[2].autoscale_view()
        self.__fig.canvas.update()
        self.__fig.canvas.flush_events()