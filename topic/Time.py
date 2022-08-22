class Time:
    def __init__(self):
        pass

    @staticmethod
    def staticFloatTimeFromTopicDict(topicRow:dict)->float:
        strSecs = str(topicRow["header"]["stamp"]["secs"])
        strNSecs = str(topicRow["header"]["stamp"]["nsecs"])
        if len(strNSecs) == 8:
            strNSecs = "0" + strNSecs
        return float(strSecs + "." + strNSecs)