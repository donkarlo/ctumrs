"""
This file is to convert topics from yaml files to python dictionaries

To use it, first replace # - - - with --- in yaml file to have a lot of yaml files in one file

This should be ran for each scenario
"""
import pickle
import string

import yaml

# To make yaml file load faster
from yaml import CLoader

from ctumrs.topics.rplidar.twoRpLidar.RangesSubTopic import RangesSubTopic


class YamlTopic:
    @staticmethod
    def getLeaderFollowerTimeRangesDictFromYaml(yamlFilePathToLidarOfTwoDronesTopic:string
                                                         , extractingDataRowsLimit = 100000)->dict:

        leaderTimeRangesObss = []
        followerTimeRangesObss = []

        with open(yamlFilePathToLidarOfTwoDronesTopic, "r") as file:
            leaderLidarCounter = 0
            followerLidarCounter = 0
            lidarDataRows = yaml.load_all(file,Loader=CLoader)
            for lidarDataCounter,lidarDataRow in enumerate(lidarDataRows):
                if leaderLidarCounter > extractingDataRowsLimit:
                    break

                robotId = lidarDataRow["header"]["frame_id"].split("/")[0]
                time = float(lidarDataRow["header"]["stamp"]["nsecs"])

                ranges = RangesSubTopic.getNpFloatedRanges(lidarDataRow["ranges"],15)

                timeRanges = [time]
                for range in ranges:
                    timeRanges.append(range)
                if robotId == "uav1":
                    leaderTimeRangesObss.append(timeRanges)
                    leaderLidarCounter += 1
                else:
                    followerTimeRangesObss.append(timeRanges)
                    followerLidarCounter += 1
        returnValue = {"leaderTimeRangesObss":leaderTimeRangesObss
                ,"followerTimeRangesObss":followerTimeRangesObss
                }
        return returnValue

if __name__ == "__main__":
    '''Load the yaml file for the two drones'''
    sharedPathToTwoLidarYaml = "/home/donkarlo/Dropbox/projs/research/data/self-aware-drones/ctumrs/two-drones/follow-scenario/lidars/"
    pathToTwoLidarsTopicYamlPath = sharedPathToTwoLidarYaml + "twoLidars.yaml"
    sharedPathToTimeRangesVel = sharedPathToTwoLidarYaml + "/"
    rtnVal = YamlTopic.getLeaderFollowerTimeRangesDictFromYaml(pathToTwoLidarsTopicYamlPath, 100000)
    pklFile = open(sharedPathToTimeRangesVel + "twoLidarsTimeRangesObss.pkl", "wb")
    pickle.dump(rtnVal, pklFile)
    pklFile.close()