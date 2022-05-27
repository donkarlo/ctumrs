from matplotlib import pyplot as plt

def replicate(data, rate):
    repData = []
    for x in data:
        for m in range(rate):
            repData.append(x)
    return repData

def plotAbnormalities(abnormalValuesScenario1
                      , abnormalValuesScenario2=None):
    # Scale the plot
    f = plt.figure()
    f.set_figwidth(10)
    f.set_figheight(2.5)

    rangeLimit = 5000
    # Label
    plt.xlabel('Timestep')
    plt.ylabel('Abnormality value')
    slicedAbnormalValuesScenario1 = abnormalValuesScenario2[0:300]+abnormalValuesScenario1[300:rangeLimit+2350]
    #abnormal scnario graph


    #normal scenario graph
    slicedAbnormalValuesScenario2 = abnormalValuesScenario2[0:rangeLimit]
    slicedAbnormalValuesScenario2 = slicedAbnormalValuesScenario1[0:2350]+slicedAbnormalValuesScenario2

    slicedAbnormalValuesScenario1 = slicedAbnormalValuesScenario1[2350:]
    slicedAbnormalValuesScenario1 =replicate(slicedAbnormalValuesScenario1,3)

    slicedAbnormalValuesScenario2 = slicedAbnormalValuesScenario2[2350:]
    slicedAbnormalValuesScenario2 = replicate(slicedAbnormalValuesScenario2, 3)


    plt.plot(range(0, rangeLimit+10000)
             , slicedAbnormalValuesScenario1
             , label=''
             , color='red'
             , linewidth=1)
    plt.plot(range(0, rangeLimit+10000)
             , slicedAbnormalValuesScenario2
             , label=''
             , color='blue'
             , linewidth=1)

    plt.ticklabel_format(axis="y"
                         , style="sci"
                         , scilimits=(0, 0))
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

normalFile = "/home/donkarlo/Desktop/lidar-autoencoder-normal-kl-10000-75-clusters-15000.txt"
followFile = "/home/donkarlo/Desktop/lidar-autoencoder-follow-kl-10000-75-clusters-15000.txt"

followAbnVals = getAbnormalityValuesFromTextFile(followFile)
normalAbnVals= getAbnormalityValuesFromTextFile(normalFile)


plotAbnormalities(followAbnVals,normalAbnVals)