from ctumrs.topic.Topic import Topic


class GpsOrigin(Topic):
    SENSOR_NAME = "gps_origin"
    def __init__(self,topicRow:dict):
        super().__init__(topicRow)

    @staticmethod
    def staticGetGpsXyz(topicRow:dict)->tuple:
        gpsX = float(topicRow["pose"]["pose"]["position"]["x"])
        gpsY = float(topicRow["pose"]["pose"]["position"]["y"])
        gpsZ = float(topicRow["pose"]["pose"]["position"]["z"])

        # gpsXVel = float(topicRow["twist"]["twist"]["linear"]["x"])
        # gpsYVel = float(topicRow["twist"]["twist"]["linear"]["y"])
        # gpsZVel = float(topicRow["twist"]["twist"]["linear"]["z"])

        return gpsX,gpsY,gpsZ