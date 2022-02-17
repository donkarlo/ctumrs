import pickle

from ctumrs.topics.rplidar.Ranges import getRangeSum
import yaml
from yaml import CLoader



def getRobotsTimeRangeSumVelObssFromYaml(yamlFilePathToLidarOfTwoDronesTopic, extractingDataRowsLimit = 100000):

    leaderRangeSumRangeVelObss = []
    leaderTimeRangeSumRangeVelObss = []
    followerRangeSumRangeVelObss = []
    followerTimeRangeSumRangeVelObss = []
    rangeVelCoefficient = 1

    with open(yamlFilePathToLidarOfTwoDronesTopic, "r") as file:
        leaderLidarCounter = 0
        followerLidarCounter = 0
        lidarDataRows = yaml.load_all(file,Loader=CLoader)
        for lidarDataCounter,lidarDataRow in enumerate(lidarDataRows):
            print(lidarDataCounter)
            if leaderLidarCounter > extractingDataRowsLimit:
                break

            robotId = lidarDataRow["header"]["frame_id"].split("/")[0]
            time = float(lidarDataRow["header"]["stamp"]["nsecs"])
            rangeSum = getRangeSum(lidarDataRow["ranges"])

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
    dataPathToLidarOfTwoDronesTopic = "/home/donkarlo/Dropbox/projs/research/data/self-aware-drones/ctumrs/two-drones/follow-scenario/lidars/"
    yamlFilePathToLidarOfTwoDronesTopic = dataPathToLidarOfTwoDronesTopic + "twoLidars.yaml"

    rtnVal = getRobotsTimeRangeSumVelObssFromYaml(yamlFilePathToLidarOfTwoDronesTopic,100000)

    pklFile = open(dataPathToLidarOfTwoDronesTopic+"twoLidarsTimeRangeSumVelObss.pkl", "wb")
    pickle.dump(rtnVal, pklFile)
    pklFile.close()