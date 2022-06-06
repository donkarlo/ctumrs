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

from ctumrs.topics.rplidar.twoRpLidar.rangeSumVel.TimeRangeSumVelObs import TimeRangeSumVelObs

class YamlTopic:
    @staticmethod
    def getLeaderFollowerTimeRangeSumVelObssDictFromYaml(yamlFilePathToLidarOfTwoDronesTopic:string
                                                         , extractingDataRowsLimit = 100000):

        leaderRangeSumRangeVelObss = []
        leaderTimeRangeSumRangeVelObss = []
        followerRangeSumRangeVelObss = []
        followerTimeRangeSumRangeVelObss = []
        rangeVelCoefficient = 10000

        with open(yamlFilePathToLidarOfTwoDronesTopic, "r") as file:
            leaderLidarCounter = 0
            followerLidarCounter = 0
            lidarDataRows = yaml.load_all(file,Loader=CLoader)
            for lidarDataCounter,lidarDataRow in enumerate(lidarDataRows):
                if leaderLidarCounter > extractingDataRowsLimit:
                    break

                robotId = lidarDataRow["header"]["frame_id"].split("/")[0]
                time = float(lidarDataRow["header"]["stamp"]["nsecs"])
                rangeSum = TimeRangeSumVelObs.getRangeSumFromListOfStringRanges(lidarDataRow["ranges"])

                if robotId == "uav1":
                    robotSpecificLidarCounter = leaderLidarCounter
                else:
                    robotSpecificLidarCounter = followerLidarCounter

                if robotSpecificLidarCounter == 0:
                    rangeSumVel = 0
                if  robotSpecificLidarCounter >= 1:
                    if robotId == "uav1":
                        prvTime = leaderTimeRangeSumRangeVelObss[robotSpecificLidarCounter-1][0]
                        prvRangeSum = leaderTimeRangeSumRangeVelObss[robotSpecificLidarCounter-1][1]
                    else:
                        prvTime = followerTimeRangeSumRangeVelObss[robotSpecificLidarCounter-1][0]
                        prvRangeSum = followerTimeRangeSumRangeVelObss[robotSpecificLidarCounter-1][1]
                    rangeSumVel = rangeVelCoefficient * (rangeSum - prvRangeSum) / (time - prvTime)
                    #set the first time sptep range sum vel
                    if robotSpecificLidarCounter == 1:
                        if robotId == "uav1":
                            leaderRangeSumRangeVelObss[0][1] = rangeSumVel
                            leaderTimeRangeSumRangeVelObss[0][2]=rangeSumVel
                        else:
                            followerRangeSumRangeVelObss[0][1] = rangeSumVel
                            followerTimeRangeSumRangeVelObss[0][2] = rangeSumVel


                if robotId == "uav1":
                    leaderRangeSumRangeVelObss.append([rangeSum, rangeSumVel])
                    leaderTimeRangeSumRangeVelObss.append([time, rangeSum, rangeSumVel])
                    leaderLidarCounter += 1
                else:
                    followerRangeSumRangeVelObss.append([rangeSum, rangeSumVel])
                    followerTimeRangeSumRangeVelObss.append([time, rangeSum, rangeSumVel])
                    followerLidarCounter += 1
        returnValue = {"leaderRangeSumVelObss":leaderRangeSumRangeVelObss
            ,"leaderTimeRangeSumVelObss":leaderTimeRangeSumRangeVelObss
                ,"followerRangeSumVelObss":followerRangeSumRangeVelObss
                ,"followerTimeRangeSumVelObss":followerTimeRangeSumRangeVelObss
                }
        return returnValue

if __name__ == "__main__":
    '''Load the yaml file for the two drones'''
    sharedPathToTwoLidarYaml = "/home/donkarlo/Dropbox/projs/research/data/self-aware-drones/ctumrs/two-drones/follow-scenario/lidars/"
    pathToTwoLidarsTopicYamlPath = sharedPathToTwoLidarYaml + "twoLidars.yaml"
    sharedPathToFourRangesVels = sharedPathToTwoLidarYaml + "rangeSumVel/"
    rtnVal = YamlTopic.getLeaderFollowerTimeRangeSumVelObssDictFromYaml(pathToTwoLidarsTopicYamlPath, 100000)
    pklFile = open(sharedPathToFourRangesVels + "twoLidarsTimeRangeSumVelObss.pkl", "wb")
    pickle.dump(rtnVal, pklFile)
    pklFile.close()