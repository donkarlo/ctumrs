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


class YamlTopic:
    @staticmethod
    def getLeaderFollowerTimeRangeSumVelObssDictFromYaml(yamlFilePathToLidarOfTwoDronesTopic:string
                                                         , extractingDataRowsLimit = 100000):

        leaderRangeSumRangeVelObss = []
        leaderTimeRangeSumRangeVelObss = []
        followerRangeSumRangeVelObss = []
        followerTimeRangeSumRangeVelObss = []

        with open(yamlFilePathToLidarOfTwoDronesTopic, "r") as file:
            leaderLidarCounter = 0
            followerLidarCounter = 0
            lidarDataRows = yaml.load_all(file,Loader=CLoader)
            for lidarDataCounter,lidarDataRow in enumerate(lidarDataRows):
                if leaderLidarCounter > extractingDataRowsLimit:
                    break

                robotId = lidarDataRow["header"]["frame_id"].split("/")[0]
                time = float(lidarDataRow["header"]["stamp"]["nsecs"])

                countRanges = len(lidarDataRow["ranges"])
                countRangesInterval = countRanges//4

                #-pi
                range1 = float(lidarDataRow["ranges"][0 * countRangesInterval])
                #-pi/2
                range2 = float(lidarDataRow["ranges"][1 * countRangesInterval])
                # 0
                range3 = float(lidarDataRow["ranges"][2 * countRangesInterval])
                # pi/2
                range4 = float(lidarDataRow["ranges"][3 * countRangesInterval])


                if robotId == "uav1":
                    robotSpecificLidarCounter = leaderLidarCounter
                else:
                    robotSpecificLidarCounter = followerLidarCounter

                if robotSpecificLidarCounter == 0:
                    range1Vel = 0
                    range2Vel = 0
                    range3Vel = 0
                    range4Vel = 0
                if  robotSpecificLidarCounter >= 1:
                    if robotId == "uav1":
                        prvTime = leaderTimeRangeSumRangeVelObss[robotSpecificLidarCounter-1][0]
                        prvRange1 = leaderTimeRangeSumRangeVelObss[robotSpecificLidarCounter-1][1]
                        prvRange2 = leaderTimeRangeSumRangeVelObss[robotSpecificLidarCounter-1][2]
                        prvRange3 = leaderTimeRangeSumRangeVelObss[robotSpecificLidarCounter-1][3]
                        prvRange4 = leaderTimeRangeSumRangeVelObss[robotSpecificLidarCounter-1][4]
                    else:
                        prvTime = followerTimeRangeSumRangeVelObss[robotSpecificLidarCounter-1][0]
                        prvRange1 = followerTimeRangeSumRangeVelObss[robotSpecificLidarCounter-1][1]
                        prvRange2 = followerTimeRangeSumRangeVelObss[robotSpecificLidarCounter-1][2]
                        prvRange3 = followerTimeRangeSumRangeVelObss[robotSpecificLidarCounter-1][3]
                        prvRange4 = followerTimeRangeSumRangeVelObss[robotSpecificLidarCounter-1][4]

                    rangeVel1 = (range1 - prvRange1) / (time - prvTime)
                    rangeVel2 = (range2 - prvRange2) / (time - prvTime)
                    rangeVel3 = (range3 - prvRange3) / (time - prvTime)
                    rangeVel4 = (range4 - prvRange4) / (time - prvTime)

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
    sharedPathToLidarYaml = "/home/donkarlo/Dropbox/projs/research/data/self-aware-drones/ctumrs/two-drones/normal-scenario/lidars/"
    sharedPathToFourRangesSum = sharedPathToLidarYaml + "fourRangesVel/"
    pathToTwoLidarsTopicYamlPath = sharedPathToLidarYaml + "twoLidars.yaml"
    rtnVal = YamlTopic.getLeaderFollowerTimeRangeSumVelObssDictFromYaml(pathToTwoLidarsTopicYamlPath, 100000)
    pklFile = open(sharedPathToFourRangesSum + "twoLidarsTimeRangeSumVelObss.pkl", "wb")
    pickle.dump(rtnVal, pklFile)
    pklFile.close()