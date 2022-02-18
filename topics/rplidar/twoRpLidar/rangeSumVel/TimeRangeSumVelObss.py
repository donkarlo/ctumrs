class TimeRangeSumVelObss:
    """This class is responsible for Position Velocity Observations"""
    @staticmethod
    def velMulInRangeSumVelObss(rangeSumVelObss:list, rangeSumVelCoefficient:float):
        '''
        '''
        for counter,element in enumerate(rangeSumVelObss):
            rangeSumVelObss[counter][1] = rangeSumVelCoefficient * rangeSumVelObss[counter][1]
        return rangeSumVelObss

    @staticmethod
    def velMulInRangeSumAndTimeRangeSumObss(rangeSumVelObss: list, timeRangeSumVelObss: list, rangeSumVelCoefficient: float):
        """"""
        for counter, element in enumerate(rangeSumVelObss):
            rangeSumVelObss[counter][1] = rangeSumVelCoefficient * rangeSumVelObss[counter][1]
            timeRangeSumVelObss[counter][2] = rangeSumVelCoefficient * timeRangeSumVelObss[counter][2]
        return [rangeSumVelObss, timeRangeSumVelObss]
