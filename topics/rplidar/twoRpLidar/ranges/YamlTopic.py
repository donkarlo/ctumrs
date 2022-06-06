"""
This file is to convert topics from yaml files to python dictionaries

To use it, first replace # - - - with --- in yaml file to have a lot of yaml files in one file

This should be ran for each scenario
"""
import pickle
import string

import numpy as np
import yaml

# To make yaml file load faster
from yaml import CLoader

from ctumrs.topics.rplidar.twoRpLidar.ranges.TimeRangesVelsObs import TimeRangesVelsObs


class YamlTopic:
    @staticmethod
    def getLeaderFollowerTimeRangesVelsObssDictFromYaml(yamlFilePathToLidarOfTwoDronesTopic:string
                                                        , extractingDataRowsLimit = 100000):

        leaderTimeRangesVelsObss = []
        followerTimeRangesVelsObss = []

        with open(yamlFilePathToLidarOfTwoDronesTopic, "r") as file:
            leaderLidarCounter = 0
            followerLidarCounter = 0
            lidarDataRows = yaml.load_all(file,Loader=CLoader)
            for lidarDataCounter,lidarDataRow in enumerate(lidarDataRows):
                if leaderLidarCounter > extractingDataRowsLimit:
                    break

                robotId = lidarDataRow["header"]["frame_id"].split("/")[0]
                time = float(lidarDataRow["header"]["stamp"]["nsecs"])
                npRanges = TimeRangesVelsObs.getNpFloatRanges(lidarDataRow["ranges"])
                rangesLen = len(npRanges)


                if robotId == "uav1":
                    robotSpecificLidarCounter = leaderLidarCounter
                else:
                    robotSpecificLidarCounter = followerLidarCounter

                if robotSpecificLidarCounter == 0:
                    npVels = np.zeros((rangesLen))
                elif  robotSpecificLidarCounter >= 1:
                    if robotId == "uav1":
                        prvTime = leaderTimeRangesVelsObss[robotSpecificLidarCounter-1][0]
                        prvNpRanges = leaderTimeRangesVelsObss[robotSpecificLidarCounter-1][1]
                    else:
                        prvTime = followerTimeRangesVelsObss[robotSpecificLidarCounter-1][0]
                        prvNpRanges = followerTimeRangesVelsObss[robotSpecificLidarCounter-1][1]

                    npVels = (npRanges - prvNpRanges) / (time - prvTime)

                    #set the first time sptep range sum vel
                    if robotSpecificLidarCounter == 1:
                        if robotId == "uav1":
                            leaderTimeRangesVelsObss[0] = np.concatenate(([prvTime],leaderTimeRangesVelsObss[0][0:rangesLen],npVels))
                        else:
                            followerTimeRangesVelsObss[0] = np.concatenate(([prvTime],followerTimeRangesVelsObss[0][0:rangesLen],npVels))


                timeRangesVels = np.concatenate(([time],npRanges,npVels))
                if robotId == "uav1":
                    leaderTimeRangesVelsObss.append(timeRangesVels)
                    leaderLidarCounter += 1
                else:
                    followerTimeRangesVelsObss.append(timeRangesVels)
                    followerLidarCounter += 1
        returnValue = {"leaderTimeRangesVelsObss":leaderTimeRangesVelsObss
                ,"followerTimeRangesVelsObss":followerTimeRangesVelsObss
                }
        return returnValue

if __name__ == "__main__":
    '''Load the yaml file for the two drones'''
    scenarioName  = "follow"
    strategyName = "ranges"

    sharedPathToTwoLidarYaml = "/home/donkarlo/Dropbox/projs/research/data/self-aware-drones/ctumrs/two-drones/{}-scenario/lidars/".format(scenarioName)
    pathToTwoLidarsTopicYamlPath = sharedPathToTwoLidarYaml + "twoLidars.yaml"
    sharedPathToFourRangesVels = sharedPathToTwoLidarYaml + "{}/".format(strategyName)
    rtnVal = YamlTopic.getLeaderFollowerTimeRangesVelsObssDictFromYaml(pathToTwoLidarsTopicYamlPath, 100000)
    pklFile = open(sharedPathToFourRangesVels + "twoLidarsTimeRangesVelsObss.pkl", "wb")
    pickle.dump(rtnVal, pklFile)
    pklFile.close()