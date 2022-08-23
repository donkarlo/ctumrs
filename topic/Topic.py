from ctumrs.topic.Time import Time


class Topic:
    def __init__(self, topicRow:dict):
        self._topic = topicRow
    def getTime(self):
        time = Time()
        return time.staticFloatTimeFromTopicDict(self._topic)
    @staticmethod
    def getSensor(topicRow:dict):
        pass

    @staticmethod
    def getRobotId(topicRow:dict):
        pass