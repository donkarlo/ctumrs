"""
This file is to convert topics from yaml files to python dictionaries

To use it, first replace # - - - with --- in yaml file to have a lot of yaml files in one file

This should be ran for each scenario
"""
import pickle
import yaml

# To make yaml file load faster
from yaml import CLoader

from ctumrs.topics.lidar.two.fourRegionsMinRangesVel.TimeFourRegionsMinRangesVelsObs import TimeFourRegionsMinRangesVelsObs


class YamlTopic:
    @staticmethod
    def getLeaderFollowerTimeRangeSumVelObssDictFromYaml(yamlFilePathToLidarOfTwoDronesTopic:str
                                                         , extractingDataRowsLimit = 100000):

        leaderFourRangesVelsObss = []
        leaderTimeFourRangesVelsObss = []
        followerFourRangesVelsObss = []
        followerTimeFourRangesVelsObss = []

        with open(yamlFilePathToLidarOfTwoDronesTopic, "r") as file:
            leaderLidarCounter = 0
            followerLidarCounter = 0
            lidarDataRows = yaml.load_all(file,Loader=CLoader)
            for lidarDataCounter,lidarDataRow in enumerate(lidarDataRows):
                if leaderLidarCounter > extractingDataRowsLimit:
                    break

                robotId = lidarDataRow["header"]["frame_id"].split("/")[0]
                time = float(lidarDataRow["header"]["stamp"]["nsecs"])

                countRanges = len(lidarDataRow["allRanges"])
                countRangesInterval = countRanges//4

                range1 = TimeFourRegionsMinRangesVelsObs.getRangesMin(lidarDataRow["allRanges"][0:1 * countRangesInterval - 1])
                range2 = TimeFourRegionsMinRangesVelsObs.getRangesMin(lidarDataRow["allRanges"][1 * countRangesInterval:2 * countRangesInterval - 1])
                range3 = TimeFourRegionsMinRangesVelsObs.getRangesMin(lidarDataRow["allRanges"][2 * countRangesInterval: 3 * countRangesInterval - 1])
                range4 = TimeFourRegionsMinRangesVelsObs.getRangesMin(lidarDataRow["allRanges"][3 * countRangesInterval:4 * countRangesInterval - 1])


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
                        prvTime = leaderTimeFourRangesVelsObss[robotSpecificLidarCounter-1][0]
                        prvRange1 = leaderTimeFourRangesVelsObss[robotSpecificLidarCounter-1][1]
                        prvRange2 = leaderTimeFourRangesVelsObss[robotSpecificLidarCounter-1][2]
                        prvRange3 = leaderTimeFourRangesVelsObss[robotSpecificLidarCounter-1][3]
                        prvRange4 = leaderTimeFourRangesVelsObss[robotSpecificLidarCounter-1][4]
                    else:
                        prvTime = followerTimeFourRangesVelsObss[robotSpecificLidarCounter-1][0]
                        prvRange1 = followerTimeFourRangesVelsObss[robotSpecificLidarCounter-1][1]
                        prvRange2 = followerTimeFourRangesVelsObss[robotSpecificLidarCounter-1][2]
                        prvRange3 = followerTimeFourRangesVelsObss[robotSpecificLidarCounter-1][3]
                        prvRange4 = followerTimeFourRangesVelsObss[robotSpecificLidarCounter-1][4]

                    range1Vel = (range1 - prvRange1) / (time - prvTime)
                    range2Vel = (range2 - prvRange2) / (time - prvTime)
                    range3Vel = (range3 - prvRange3) / (time - prvTime)
                    range4Vel = (range4 - prvRange4) / (time - prvTime)

                    #set the first time sptep range sum vel
                    if robotSpecificLidarCounter == 1:
                        if robotId == "uav1":
                            leaderFourRangesVelsObss[0][4] = range1Vel
                            leaderFourRangesVelsObss[0][5] = range2Vel
                            leaderFourRangesVelsObss[0][6] = range3Vel
                            leaderFourRangesVelsObss[0][7] = range4Vel
                            
                            leaderTimeFourRangesVelsObss[0][5]=range1Vel
                            leaderTimeFourRangesVelsObss[0][6]=range2Vel
                            leaderTimeFourRangesVelsObss[0][7]=range3Vel
                            leaderTimeFourRangesVelsObss[0][8]=range4Vel
                        else:
                            followerFourRangesVelsObss[0][4] = range1Vel
                            followerFourRangesVelsObss[0][5] = range2Vel
                            followerFourRangesVelsObss[0][6] = range3Vel
                            followerFourRangesVelsObss[0][7] = range4Vel

                            followerTimeFourRangesVelsObss[0][5] = range1Vel
                            followerTimeFourRangesVelsObss[0][6] = range2Vel
                            followerTimeFourRangesVelsObss[0][7] = range3Vel
                            followerTimeFourRangesVelsObss[0][8] = range4Vel


                fourRangesVel = [range1
                                 ,range2
                                 ,range3
                                 ,range4
                                 ,range1Vel
                                 , range2Vel
                                 , range3Vel
                                 , range4Vel
                                 ]

                timeFourRangesVel = [time
                    ,range1
                    , range2
                    , range3
                    , range4
                    , range1Vel
                    , range2Vel
                    , range3Vel
                    , range4Vel
                                 ]
                if robotId == "uav1":
                    leaderFourRangesVelsObss.append(fourRangesVel)
                    leaderTimeFourRangesVelsObss.append(timeFourRangesVel)
                    leaderLidarCounter += 1
                else:
                    followerFourRangesVelsObss.append(fourRangesVel)
                    followerTimeFourRangesVelsObss.append(timeFourRangesVel)
                    followerLidarCounter += 1
        returnValue = {"leaderFourRegionsMinRangesVelsObss":leaderFourRangesVelsObss
            ,"leaderTimeFourRegionsMinRangesVelsObss":leaderTimeFourRangesVelsObss
                ,"followerFourRegionsMinRangesVelsObss":followerFourRangesVelsObss
                ,"followerTimeFourRegionsMinRangesVelsObss":followerTimeFourRangesVelsObss
                }
        return returnValue

if __name__ == "__main__":
    '''
        Load the yaml file for the two drones
        - first load the paths to normal scenarios
        - Build the transition matrix for normal scenario booter
        - second load the paths to abnormal scenarios
        - run novelty for that abnormal scenario
        @todo make it such that each scenario can put its own path
    '''
    sharedPathToTwoLidarYaml = "/home/donkarlo/Dropbox/projs/research/data/self-aware-drones/ctumrs/two-drones/follow-scenario/lidars/"
    pathToTwoLidarsTopicYamlPath = sharedPathToTwoLidarYaml + "twoLidars.yaml"
    sharedPathToFourRangesVels = sharedPathToTwoLidarYaml + "fourRegionsMinRangesVels/"
    rtnVal = YamlTopic.getLeaderFollowerTimeRangeSumVelObssDictFromYaml(pathToTwoLidarsTopicYamlPath, 100000)
    pklFile = open(sharedPathToFourRangesVels + "twoLidarsTimeFourRegionsMinRangesVelsObss.pkl", "wb")
    pickle.dump(rtnVal, pklFile)
    pklFile.close()
    print("Yaml changed to pkl")