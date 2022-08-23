class Time:
    def __init__(self):
        pass

    @staticmethod
    def staticFloatTimeFromTopicDict(topicRow:dict)->float:
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
        return float(strSecs + "." + strNSecs)