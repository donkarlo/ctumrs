from matplotlib import pyplot as plt
def plotAbnormalities(abnormalValuesScenario1
                      , abnormalValuesScenario2=None):
    # Scale the plot
    f = plt.figure()
    f.set_figwidth(10)
    f.set_figheight(2.5)

    rangeLimit = 15000
    # Label
    plt.xlabel('Timestep')
    plt.ylabel('Abnormality value')
    plt.plot(range(0, rangeLimit)
             , abnormalValuesScenario1[0:rangeLimit]
             , label=''
             , color='blue'
             , linewidth=1)

    if abnormalValuesScenario2 is not None:
        plt.plot(range(0, rangeLimit)
                 , abnormalValuesScenario2[0:rangeLimit]
                 , label=''
                 , color='red'
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
        if float(line.strip()) > 500000:
            print (count, " ",float(line.strip()))
        abnVal = 0 if float(line.strip())>1000000000 else float(line.strip())
        abnValues.append(abnVal)
    file.close()
    return abnValues

normalFile = "/home/donkarlo/Desktop/lidar-normal-abnormality-values-kl-velco-20-clusters-75-75-timesteps-15000.txt"
followFile = "/home/donkarlo/Desktop/lidar-follow-abnormality-values-kl-velco-20-clusters-75-75-timesteps-15000.txt"

normalAbnVals= getAbnormalityValuesFromTextFile(normalFile)
followAbnVals = getAbnormalityValuesFromTextFile(followFile)


plotAbnormalities(normalAbnVals)