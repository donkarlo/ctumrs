import numpy as np


class TimeRangesVelsObss:
    """This class is responsible for Position Velocity Observations"""
    @staticmethod
    def velMulInTimeRangesVelsObss(npTimeRangesVelsObss: np.ndarray, coefficient: float):
        """"""
        velStartIdx = int((len(npTimeRangesVelsObss[0]) - 1) / 2 + 1)
        npTimeRangesVelsObss[:, velStartIdx:] *= coefficient
        return npTimeRangesVelsObss
