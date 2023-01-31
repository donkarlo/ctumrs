import pickle

import matplotlib
import numpy as np
from matplotlib import pyplot as plt


class Abnormality:
    def __init__(self):
        pass

    @staticmethod
    def staticSaveGps(filePathName:str,timeAbnVals:list):
        with open(filePathName, 'wb') as filehandler:
            pickle.dump(timeAbnVals, filehandler)

    @staticmethod
    def staticSaveLidar(filePathName:str,timeAbnVals:list):
        with open(filePathName, 'wb') as filehandler:
            pickle.dump(timeAbnVals, filehandler)

    @staticmethod
    def loadTimeAbnVals(filePathName)->pickle:
        with open(filePathName, 'rb') as file:
            loadedPickle = pickle.load(file)
            return loadedPickle

    @staticmethod
    def plotNormalConfidenceLineAbnormalValsAllTogether(normalTestTimeAbnValsList:list):
        matplotlib.use("Qt5Agg")
        # plt.gca().set_aspect('equal')
        for normalTimeTestAbnVals in reversed(normalTestTimeAbnValsList):
            normalAbnMean3Sigma = np.mean(normalTimeTestAbnVals[0][:,1]) + 3*np.std(normalTimeTestAbnVals[0][:,1])
            plt.plot(normalTimeTestAbnVals[1][:,0]#time
                     ,normalTimeTestAbnVals[1][:,1]#abnVals
                     ,label=normalTimeTestAbnVals[2])

            plt.plot([normalTimeTestAbnVals[1][0,0], normalTimeTestAbnVals[1][-1,0]]
                     , [normalAbnMean3Sigma, normalAbnMean3Sigma]
                     , label=normalTimeTestAbnVals[2])

        plt.legend()
        plt.show()

    @staticmethod
    def plotAbnValsTolerenceLines(gpsNormalTestTimeAbnValsList:list, lidarNormalTestTimeAbnValsList:list):
        matplotlib.use("Qt5Agg")
        plotColsNum = max (len(gpsNormalTestTimeAbnValsList),len(lidarNormalTestTimeAbnValsList))
        fig, axes = plt.subplots(plotColsNum, 2)

        for plotRowCounter in range(plotColsNum):
            #GPS
            gpsNormalAbnMeanMulSigma = np.mean(gpsNormalTestTimeAbnValsList[plotRowCounter][0][:, 1]) + 3 * np.std(gpsNormalTestTimeAbnValsList[plotRowCounter][0][:, 1])
            axes[plotRowCounter][0].plot(gpsNormalTestTimeAbnValsList[plotRowCounter][1][:,0]
                                         ,gpsNormalTestTimeAbnValsList[plotRowCounter][1][:,1]
                                         ,label = gpsNormalTestTimeAbnValsList[plotRowCounter][2]
                                         ,linewidth='1'
                                         )
            axes[plotRowCounter][0].plot([gpsNormalTestTimeAbnValsList[plotRowCounter][1][0,0], gpsNormalTestTimeAbnValsList[plotRowCounter][1][-1,0]]
                                         ,[gpsNormalAbnMeanMulSigma,gpsNormalAbnMeanMulSigma]
                                         )
            axes[plotRowCounter][0].tick_params(axis='both', labelsize=8)
            axes[plotRowCounter][0].legend(fontsize=8)


            #LIDAR
            lidarNormalAbnMeanMulSigma = np.mean(lidarNormalTestTimeAbnValsList[plotRowCounter][0][:, 1]) + 1 * np.std(lidarNormalTestTimeAbnValsList[plotRowCounter][0][:, 1])
            axes[plotRowCounter][1].plot(lidarNormalTestTimeAbnValsList[plotRowCounter][1][:,0]
                                         ,lidarNormalTestTimeAbnValsList[plotRowCounter][1][:,1]
                                         , label=lidarNormalTestTimeAbnValsList[plotRowCounter][2]
                                         , linewidth='1'
                                         )
            axes[plotRowCounter][1].plot([lidarNormalTestTimeAbnValsList[plotRowCounter][1][0, 0], lidarNormalTestTimeAbnValsList[plotRowCounter][1][-1, 0]]
                                         , [lidarNormalAbnMeanMulSigma, lidarNormalAbnMeanMulSigma]
                                         )

            axes[plotRowCounter][1].tick_params(axis='both', labelsize=8)
            axes[plotRowCounter][1].legend(fontsize=8)
        plt.show()



















