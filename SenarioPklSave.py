import os
import pickle

import numpy as np

from MachineSettings import MachineSettings
import yaml
from yaml import CLoader
from ctumrs.topic.GpsOrigin import GpsOrigin
from ctumrs.topic.RpLidar import RpLidar
from ctumrs.topic.Topic import Topic

# configs
with open("configs.yaml", "r") as file:
    configs = yaml.load(file, Loader=CLoader)
testSharedPath = "/home/donkarlo/Desktop/lstm/"

class ScenarioPklSave:
    def savePkl(self):
        scenarioName = "follow"
        basePath = MachineSettings.MAIN_PATH + "projs/research/data/self-aware-drones/ctumrs/two-drones/"
        pathToScenario = basePath + "{}-scenario/".format(scenarioName)

        pathToNormalScenarioYamlFile = pathToScenario + "uav1-gps-lidar-uav2-gps-lidar.yaml"

        # Lidar settings
        lidarSensorName = "rplidar"

        # Gps settings
        gpsSensorName = "gps_origin"

        topicCounter = 0
        with open(pathToNormalScenarioYamlFile, "r") as file:
            topicRows = yaml.load_all(file, Loader=CLoader)

            robotIdTimeSensorObss = []

            for topicRowCounter, topicRow in enumerate(topicRows):
                robotId, sensorName = Topic.staticGetRobotIdAndSensorName(topicRow)
                time = Topic.staticGetTimeByTopicDict(topicRow)
                if sensorName == gpsSensorName:
                    gpsX, gpsY, gpsZ = np.array(GpsOrigin.staticGetGpsXyz(topicRow))
                    robotIdTimeSensorObss.append({"time":time, "robotId":robotId,"sensorName":sensorName,"npValue":np.array([gpsX, gpsY, gpsZ])})

                if sensorName == lidarSensorName:
                    npRanges = RpLidar.staticGetNpRanges(topicRow)
                    robotIdTimeSensorObss.append({"time": time, "robotId":robotId,"sensorName": sensorName, "npValue": npRanges})
                # if topicRowCounter > 5000:
                #     break
                print(topicRowCounter)
        with open('{}/{}ScenarioRobotIdTimeSensorObss.pkl'.format(testSharedPath,scenarioName), 'wb') as file:
            pickle.dump(robotIdTimeSensorObss, file)

if __name__=="__main__":
    scenarioPklSave = ScenarioPklSave()
    scenarioPklSave.savePkl()
