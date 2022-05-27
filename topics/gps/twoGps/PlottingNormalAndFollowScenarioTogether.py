from matplotlib import pyplot as plt


def plotAbnormalities(abnormalValuesScenario1, abnormalValuesScenario2=None):
    # Scale the plot
    f = plt.figure()
    f.set_figwidth(10)
    f.set_figheight(2.5)

    rangeLimit = 15000
    # Label
    plt.xlabel('Timestep')
    plt.ylabel('Abnormality value')
    slicedAbnormalValuesScenario1 = abnormalValuesScenario1[0:rangeLimit]
    plt.plot(range(0, rangeLimit)
             , slicedAbnormalValuesScenario1
             , label=''
             , color='red'
             , linewidth=1)

    if abnormalValuesScenario2 is not None:
        slicedAbnormalValuesScenario2 = abnormalValuesScenario2[0:rangeLimit]
        plt.plot(range(0, rangeLimit)
                 , slicedAbnormalValuesScenario2
                 , label=''
                 , color='blue'
                 , linewidth=1)
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
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

normalFile = "/home/donkarlo/Desktop/gps-normal-kl-10000-75-clusters-15000.txt"
followFile = "/home/donkarlo/Desktop/gps-follow-kl-10000-75-clusters-15000.txt"

followAbnVals = getAbnormalityValuesFromTextFile(followFile)
normalAbnVals= getAbnormalityValuesFromTextFile(normalFile)


plotAbnormalities(followAbnVals,normalAbnVals)