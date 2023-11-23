import numpy as np

from ctumrs.topic.Topic import Topic
class RpLidar(Topic):
    SENSOR_NAME = "rplidar"
    def __init__(self,topicRow):
        super().__init__(topicRow)

    @staticmethod
    def staticGetNpRanges(topicRow:dict, infReplacement=15):
        npRanges = np.array(topicRow["ranges"]).astype(float)
        # replace infs with 15 in np.range
        npRanges[npRanges == np.inf] = infReplacement
        return npRanges