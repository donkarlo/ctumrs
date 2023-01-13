import pickle


class Abnormality:
    def __init__(self):
        pass

    @staticmethod
    def staticSaveGps(filePathName:str,timeAbnVals:list):
        with open(filePathName, 'wb') as filehandler:
            pickle.dump(timeAbnVals, filehandler)

    @staticmethod
    def staticSaveLidar(filePathName:str,timeAbnVals:list):
        with open(filePathName, 'wb') as filehandler:
            pickle.dump(timeAbnVals, filehandler)
