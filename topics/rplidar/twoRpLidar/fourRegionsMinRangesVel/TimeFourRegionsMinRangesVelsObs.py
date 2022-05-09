class TimeFourRegionsMinRangesVelsObs:
    @staticmethod
    def getRangesMin(strRanges: str):
        minFloatRange = 15
        for curStrRange in strRanges:
            if curStrRange == "inf":
                curFloatRange = 15
            else:
                curFloatRange = float(curStrRange)
            if curFloatRange < minFloatRange:
                minFloatRange = curFloatRange
        return minFloatRange
