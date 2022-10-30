import numpy as np


class Topic:
    def __init__(self, topicRow:dict):
        self._topic = topicRow

    @staticmethod
    def staticGetRobotIdAndSensorName(topicRow:dict):
        robotId,sensorName = topicRow["header"]["frame_id"].split("/")
        return robotId,sensorName

    @staticmethod
    def staticGetTimeByTopicDict(topicRow: dict) -> float:
        strSecs = str(topicRow["header"]["stamp"]["secs"])
        strNSecs = str(topicRow["header"]["stamp"]["nsecs"])

        if len(strNSecs) == 8:
            strNSecs = "0" + strNSecs
        elif len(strNSecs) == 7:
            strNSecs = "00" + strNSecs
        elif len(strNSecs) == 6:
            strNSecs = "000" + strNSecs
        elif len(strNSecs) == 5:
            strNSecs = "0000" + strNSecs
        elif len(strNSecs) == 4:
            strNSecs = "00000" + strNSecs
        elif len(strNSecs) == 3:
            strNSecs = "000000" + strNSecs
        elif len(strNSecs) == 2:
            strNSecs = "0000000" + strNSecs
        elif len(strNSecs) == 1:
            strNSecs = "00000000" + strNSecs

        strTime = strSecs + "." + strNSecs
        return np.float64(strTime)