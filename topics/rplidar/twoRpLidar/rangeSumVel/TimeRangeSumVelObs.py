import string


class TimeRangeSumVelObs:
    @staticmethod
    def getRangeSumFromListOfStringRanges(ranges: list[string]):
        rangeSum = 0
        for range in (ranges):
            if range == "inf":
                range = 100
            rangeSum += float(range)
        return rangeSum