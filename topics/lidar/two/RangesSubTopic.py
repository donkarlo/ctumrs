import numpy as np


class RangesSubTopic:
    @staticmethod
    def getNpFloatedRanges(strRanges:list,infSubstitutionValue:float=15)->np.ndarray:
        floatedRanges = []
        for strRange in strRanges:
            floatedRanges.append(float(strRange) if strRange!="inf" else infSubstitutionValue)
        return floatedRanges


