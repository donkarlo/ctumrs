import numpy as np


class TimeRangesVelsObs:
    @staticmethod
    def getNpFloatRanges(strRanges: str):
        floatRanges = []
        for strRange in strRanges:
            if strRange == "inf":
                floatRange = 15
            else:
                floatRange = float(strRange)
            floatRanges.append(floatRange)
        return np.array(floatRanges)