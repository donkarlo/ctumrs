class TimeFourRangesVelsObss:
    """This class is responsible for Position Velocity Observations"""
    @staticmethod
    def velMulInRangeSumVelObss(fourRangesVelsObss:list, coefficient:float):
        '''
        '''
        for counter,element in enumerate(fourRangesVelsObss):
            fourRangesVelsObss[counter][4] = coefficient * fourRangesVelsObss[counter][4]
            fourRangesVelsObss[counter][5] = coefficient * fourRangesVelsObss[counter][5]
            fourRangesVelsObss[counter][6] = coefficient * fourRangesVelsObss[counter][6]
            fourRangesVelsObss[counter][7] = coefficient * fourRangesVelsObss[counter][7]
        return fourRangesVelsObss

    @staticmethod
    def velMulInFourRangesVelsAndTimeFourRangesVelsObss(fourRangesVelsObss: list, timeFourRangesVelsObss: list, coefficient: float):
        """"""
        for counter, element in enumerate(fourRangesVelsObss):
            fourRangesVelsObss[counter][4] = coefficient * fourRangesVelsObss[counter][4]
            fourRangesVelsObss[counter][5] = coefficient * fourRangesVelsObss[counter][5]
            fourRangesVelsObss[counter][6] = coefficient * fourRangesVelsObss[counter][6]
            fourRangesVelsObss[counter][7] = coefficient * fourRangesVelsObss[counter][7]


            timeFourRangesVelsObss[counter][5] = coefficient * timeFourRangesVelsObss[counter][5]
            timeFourRangesVelsObss[counter][6] = coefficient * timeFourRangesVelsObss[counter][6]
            timeFourRangesVelsObss[counter][7] = coefficient * timeFourRangesVelsObss[counter][7]
            timeFourRangesVelsObss[counter][8] = coefficient * timeFourRangesVelsObss[counter][8]
        return [fourRangesVelsObss, timeFourRangesVelsObss]
