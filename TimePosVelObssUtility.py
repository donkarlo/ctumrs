import numpy as np


class TimePosVelObssUtility():
    def __init__(self):
        pass

    @staticmethod
    def getTimePosVelsAndPosVels(pathToTimePosVelDataFile: str, velocityCoefficient: float) -> None:
        timePosVelDataFile = open(pathToTimePosVelDataFile, 'r')
        timePosVelDataFileLines = timePosVelDataFile.readlines()

        # Strips the newline character
        uavPosVels = []
        uavTimePosVels = []
        for timePosVelDataFileLineCounter, timePosVelDataFileLine in enumerate(timePosVelDataFileLines):
            if timePosVelDataFileLineCounter <= 100000:
                uavStrTime, \
                uavStrPosX, \
                uavStrPosY, \
                uavStrPosZ, \
                uavStrVelX, \
                uavStrVelY, \
                uavStrVelZ = timePosVelDataFileLine.strip().split(",")

                uavTime = float(uavStrTime)

                uavPosX = float(uavStrPosX)
                uavPosY = float(uavStrPosY)
                uavPosZ = float(uavStrPosZ)

                uavVelX = velocityCoefficient * float(uavStrVelX)
                uavVelY = velocityCoefficient * float(uavStrVelY)
                uavVelZ = velocityCoefficient * float(uavStrVelZ)

                uavPosVels.append([uavPosX
                                      , uavPosY
                                      , uavPosZ
                                      , uavVelX
                                      , uavVelY
                                      , uavVelZ])

                uavTimePosVels.append([uavTime
                                          , uavPosX
                                          , uavPosY
                                          , uavPosZ
                                          , uavVelX
                                          , uavVelY
                                          , uavVelZ])

        uavPosVels = np.array(uavPosVels)
        uavTimePosVels = np.array(uavTimePosVels)
        return {'posVels': uavPosVels, 'timePosVels': uavTimePosVels}

    def getPosVelByTimePosVel(self, timePosVel: list) -> list:
        return timePosVel[1:]

    @staticmethod
    def findClosestTimeWiseFollowerTimePosVelToLeaderTimePosVel(leaderTimePosVelObs
                                                                , followerTimePosVelObss) -> list:
        start = 0
        end = len(followerTimePosVelObss) - 1

        while end - start >= 3:
            mid = start + int((end - start) / 2)
            if leaderTimePosVelObs[0] == followerTimePosVelObss[mid][0]:
                return followerTimePosVelObss[mid]
            elif leaderTimePosVelObs[0] > followerTimePosVelObss[mid][0]:
                start = mid
                end = end
            elif leaderTimePosVelObs[0] < followerTimePosVelObss[mid][0]:
                start = start
                end = mid
        return followerTimePosVelObss[end - 1]