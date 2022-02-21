import string


class TimeFourRangesVelsObs:
    @staticmethod
    def getFloatRange(strRange: str):
        if strRange == "inf":
            return 15
        else:
            return float(strRange)