import pickle

import numpy as np
import yaml
from matplotlib import pyplot as plt
from yaml import CLoader

from ctumrs.sensors.liveLocSensorAbn.two.PlotPosGpsLidarLive import PlotPosGpsLidarLive

testSharedPath = "/home/donkarlo/Desktop/lstm/"
# configs
with open("configs.yaml", "r") as file:
    configs = yaml.load(file, Loader=CLoader)
testSharedPath = "/home/donkarlo/Desktop/lstm/"
numOfRobots = len(configs["targetRobotIds"])


if __name__=="__main__":
    normalScenarioStartTime = configs["normalScenarioStartTime"]
    # GPS settings
    robot1GpsCounter = 0
    robot2GpsCounter = 0
    gpsTimeAbnVals = []
    npRobot1TimeGpsValVelObss = []
    npRobot2TimeGpsValVelObss = []

    # Lidar settings
    lidarTestScenarioCounterLimit = configs["lidarTestScenarioCounterLimit"]
    robot1LidarCounter = 0
    robot2LidarCounter = 0
    lidarTimeAbnVals = []


    gpsExpectedHighAbnTimeInterval = (110, 160)
    lidarExpectedHighAbnTimeInterval = (75, 200)

    with open('{}/latent-dim2-training-36000-velco-2-new-encoder-best/followScenarioGpsTimeAbnVals.pkl'.format(testSharedPath), 'rb') as file:
        npTimeGpsAbnVals = np.array(pickle.load(file))

    with open('{}/latent-dim2-training-36000-velco-2-new-encoder-best/followScenarioLidarTimeAbnVals.pkl'.format(testSharedPath), 'rb') as file:
        npTimeLidarAbnVals = np.array(pickle.load(file))

    plotAll = PlotPosGpsLidarLive()
    with open('{}/followScenarioRobotIdTimeSensorObss.pkl'.format(testSharedPath), 'rb') as file:
        topicRows = pickle.load(file)[0:25000]

        beginningSkipCounter = 0
        # loop through topics
        for topicRowCounter, topicRow in enumerate(topicRows):
            if beginningSkipCounter < configs["beginningSkip"]:
                beginningSkipCounter += 1
                continue
            # How far should I go?
            if robot1LidarCounter >= lidarTestScenarioCounterLimit:
                break

            robotId, sensorName = topicRow["robotId"], topicRow["sensorName"]
            time = topicRow["time"] - normalScenarioStartTime
            npValObs = topicRow["npValue"].tolist()

            if sensorName == "gps_origin":
                if robotId == configs["targetRobotIds"][0]:
                    npRobot1TimeGpsValVelObss.append([time]+ npValObs)
                elif robotId == configs["targetRobotIds"][1]:
                    npRobot2TimeGpsValVelObss.append([time]+ npValObs)
                if(len(npRobot1TimeGpsValVelObss)>0 and len(npRobot2TimeGpsValVelObss)>0):
                    plotAll.updateGpsPlot(np.asarray(npRobot1TimeGpsValVelObss),
                                          np.asarray(npRobot2TimeGpsValVelObss))

                gpsIndex = np.searchsorted(npTimeGpsAbnVals[:, 0], time, side='right')
                plotAll.updateGpsAbnPlot(npTimeGpsAbnVals[:gpsIndex])
            if sensorName=="rplidar":
                lidarIndex = np.searchsorted(npTimeLidarAbnVals[:, 0], time, side='right')
                plotAll.updateLidarAbnPlot(npTimeLidarAbnVals[:lidarIndex])
                pass