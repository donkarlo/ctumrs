import numpy as np


class Noise:
    @staticmethod
    def addSymetricGaussianNoiseToAVec(npVec:np.ndarray,var:float):
        """"""
        noise = np.random.normal(0, var, (npVec.shape))
        return npVec+noise

if __name__ == "__main__":
    a,b,c =  Noise.addSymetricGaussianNoiseToAVec(np.array([3,2,3]),0.1)
    print(a,b,c)
