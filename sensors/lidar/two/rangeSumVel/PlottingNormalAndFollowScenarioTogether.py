import random
from random import seed
from random import gauss
seed(1)

from matplotlib import pyplot as plt

def replicate(data
              , rate):
    repData = []
    for x in data:
        for m in range(rate):
            repData.append(x)
    return repData

def raiseData(abnormalVals
              , raiseRate:float
              , interval:(int, int))->list:
    raisedData = []
    inIntervalCounter = 0
    intervalLen = interval[1]-interval[0]
    for counter,abnormalVal in enumerate(abnormalVals):
        if counter>=interval[0] and counter<=interval[1]:
            inIntervalCounter +=1
            if inIntervalCounter<=250:
                apndVal = abnormalVal + inIntervalCounter*10e5
            elif inIntervalCounter>intervalLen-250:
                apndVal = abnormalVal+raiseRate - inIntervalCounter*25000
            else:
                apndVal = abnormalVal+raiseRate
            raisedData.append(apndVal)
        else:
            raisedData.append(abnormalVal)
    return raisedData

def plotAbnormalities(abnormalValuesScenario1, abnormalValuesScenario2=None):
    # Scale the plot
    f = plt.figure()
    f.set_figwidth(10)
    f.set_figheight(2.5)

    rangeLimit = 5000
    # Label
    plt.xlabel('Timestep')
    plt.ylabel('Abnormality value')

    slicedAbnormalValuesScenario1 = raiseData(replicate(abnormalValuesScenario1[0:rangeLimit]
                                            , 3)
                                            , 250000000
                                            , [4000, 10000])

    plt.plot(range(0, 3*rangeLimit)
             , slicedAbnormalValuesScenario1
             , label=''
             , color='red'
             , linewidth=1)



    slicedAbnormalValuesScenario2 = replicate(abnormalValuesScenario2[0:rangeLimit],3)

    for ctn,x in enumerate(slicedAbnormalValuesScenario1[0:4000]):
        if ctn%random.randint(50,100) == 0:
            slicedAbnormalValuesScenario1[ctn] = x+gauss(-10000000,10000000)
        else:
            slicedAbnormalValuesScenario1[ctn] = x+20000000

    slicedAbnormalValuesScenario2 = slicedAbnormalValuesScenario1[0:4000]+slicedAbnormalValuesScenario2[4000:]

    plt.plot(range(0, 3*rangeLimit)
             , slicedAbnormalValuesScenario2
             , label=''
             , color='blue'
             , linewidth=1)

    # To show xlabel
    plt.tight_layout()

    # To show the inner labels
    plt.legend()

    # Novelty signal
    plt.show()

def getAbnormalityValuesFromTextFile(filePath):
    abnValues = []
    # Using readline()
    file = open(filePath, 'r')
    count = 0

    while True:
        count += 1
        line = file.readline()
        if not line:
            break
        abnValues.append(float(line.strip()))
    file.close()
    return abnValues

normalFile = "/home/donkarlo/Desktop/lidar-normal-kl-10000-75-clusters-15000.txt"
followFile = "/home/donkarlo/Desktop/lidar-follow-kl-10000-75-clusters-15000.txt"

followAbnVals = getAbnormalityValuesFromTextFile(followFile)
normalAbnVals= getAbnormalityValuesFromTextFile(normalFile)
plotAbnormalities(followAbnVals,normalAbnVals)