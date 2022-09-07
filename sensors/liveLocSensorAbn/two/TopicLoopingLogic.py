import os


class TopicLoopingLogic:
    @staticmethod
    def getShouldLoopThroughTopics(pathToGpsTwoAlphTransMtxFile,pathToLidarTwoAlphaTransMtxFile) -> bool:
        return not os.path.exists(pathToGpsTwoAlphTransMtxFile) \
               or not os.path.exists(pathToLidarTwoAlphaTransMtxFile)