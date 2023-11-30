import pickle

import numpy as np
from matplotlib import pyplot as plt

testSharedPath = "/home/donkarlo/Desktop/lstm/"

class ThresholdFinder:
    def __init__(self,sensorName,npTimeAbnVals:np.ndarray,highAbnTimeInterval:tuple,minMaxThresholdDivNum:int):
        self.__sensorName = sensorName
        self._npTimeAbnVals = npTimeAbnVals
        self._highAbnTimeInterval = highAbnTimeInterval
        self._minMaxThresholdDivNum = minMaxThresholdDivNum

    def getBestMLParams(self):
        minHeight = 0
        maxHeight = np.max(self._npTimeAbnVals[:,1])
        step = (maxHeight-minHeight)/self._minMaxThresholdDivNum
        thresholdVals = np.arange(minHeight, maxHeight, step)
        thresholdValPrRecF1DictList = []

        tpVals = []
        tnVals = []
        fpVals = []
        fnVals = []
        for thresholdVal in thresholdVals:
            tp = 0
            tn = 0
            fp = 0
            fn = 0
            for npTimeAbnVal in self._npTimeAbnVals:
                time = npTimeAbnVal[0]
                abnVal = npTimeAbnVal[1]
                if self._highAbnTimeInterval[0]<time<self._highAbnTimeInterval[1]:
                    if abnVal>thresholdVal:#above the line
                        tp += 1# it is correctly classified as a high abnormal
                    else:#under the line
                        fn += 1 # it is incorrectly classified as a not abnormal
                else:
                    if abnVal<thresholdVal:
                        tn += 1# it is correctly classified as a not abnormal
                    else:
                        fp += 1# it is incorrectly classified as a high abnormal
            tpVals.append(tp)
            tnVals.append(tn)
            fpVals.append(fp)
            fnVals.append(fn)
            # Calculate precision, recall, and F1 score
            precision = tp / (tp + fp) if tp + fp > 0 else 0
            recall = tp / (tp + fn) if tp + fn > 0 else 0
            f1Score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
            thresholdValPrRecF1DictList.append({"threshold":thresholdVal,"precsion":precision,"recall":recall,"f1Score":f1Score})
            # print(f"Thershold: {thresholdVal}, Precision: {precision}, recall: {recall}, f1Score: {f1}")

        # Access the corresponding value in the first column
        modelAssessParamsDict = max(thresholdValPrRecF1DictList, key=lambda x: x["f1Score"])

        #plot ROC curve
        # Calculate the true positive rate (sensitivity or recall)
        tprVals = [tp / (tp + fn) for tp, fn in zip(tpVals, fnVals)]

        # Calculate the false positive rate
        fprVals = [fp / (fp + tn) for fp, tn in zip(fpVals, tnVals)]

        # Plot ROC curve
        plt.figure()
        plt.plot(fprVals, tprVals, color='darkorange', lw=2, label='ROC curve')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{self.__sensorName} Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.show()

        # Find the best threshold based on the ROC curve
        rocCurvePoints = list(zip(fprVals, tprVals, thresholdVals))
        # Choose based on maximizing TPR - FPR
        bestRocPoint = max(rocCurvePoints, key=lambda x: x[1] - x[0])
        bestRocThreshold = bestRocPoint[2]

        modelAssessParamsDict["bestROCThreshold"]= bestRocThreshold

        return modelAssessParamsDict

if __name__=="__main__":
    with open('{}/followScenarioGpsTimeAbnVals.pkl'.format(testSharedPath), 'rb') as file:
        npTimeAbnVals = np.array(pickle.load(file))
    gpsTrFinder = ThresholdFinder("GPS", npTimeAbnVals, (110, 160), 1000)
    print(f"Best Gps threshold value: {gpsTrFinder.getBestMLParams()}")


    with open('{}/followScenarioLidarTimeAbnVals.pkl'.format(testSharedPath), 'rb') as file:
        npTimeAbnVals = np.array(pickle.load(file))
    lidarTrFinder = ThresholdFinder("LIDAR", npTimeAbnVals, (75, 200), 1000)
    print(f"Best Lidar threshold value: {lidarTrFinder.getBestMLParams()}")
