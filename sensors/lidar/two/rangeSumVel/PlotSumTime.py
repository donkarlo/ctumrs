from matplotlib import pyplot as plt
import pickle

class PlotSumTime:

    @staticmethod
    def plotNormalLeaderSumTime():
        jointPathToLeaderAndFollowerNormalScenario = "/home/donkarlo/Dropbox/projs/research/data/self-aware-drones/ctumrs/two-drones/normal-scenario/lidar/rangeSumVel/"
        pklFile = open(jointPathToLeaderAndFollowerNormalScenario + "twoLidarsTimeRangeSumVelObss.pkl", "rb")
        leaderFollowerTimeRangeSumVelDict = pickle.load(pklFile)

        leaderPosVelObss = leaderFollowerTimeRangeSumVelDict['leaderRangeSumVelObss']
        leaderSumValues = []
        for leaderPosVelObs in leaderPosVelObss:
            leaderSumValues.append(leaderPosVelObs[0])

        followerPosVelObss = leaderFollowerTimeRangeSumVelDict['followerRangeSumVelObss']
        followerSumValues = []
        for followerPosVelObs in followerPosVelObss:
            followerSumValues.append(followerPosVelObs[0])

        # Scale the plot
        f = plt.figure()
        f.set_figwidth(20)
        f.set_figheight(5)
        # Label
        oneCycleLength = 5000
        plt.xlabel('Timestep')
        plt.ylabel('Leader and follower sum of allRanges')
        plt.plot(range(0, oneCycleLength)
                 , leaderSumValues[0:oneCycleLength]
                 , label="Leader"
                 , color='red'
                 , linewidth=1)

        plt.plot(range(0, oneCycleLength)
                 , followerSumValues[0:oneCycleLength]
                 , label="Follower"
                 , color='green'
                 , linewidth=1)
        # To show xlabel
        plt.tight_layout()

        # To show the inner labels
        plt.legend("Leader sum of allRanges, normal")

        # Novelty signal
        plt.show()

    @staticmethod
    def plotFollowLeaderSumTime():
        jointPathToLeaderAndFollowerNormalScenario = "/home/donkarlo/Dropbox/projs/research/data/self-aware-drones/ctumrs/two-drones/follow-scenario/lidar/rangeSumVel/"
        pklFile = open(jointPathToLeaderAndFollowerNormalScenario + "twoLidarsTimeRangeSumVelObss.pkl", "rb")
        leaderFollowerTimeRangeSumVelDict = pickle.load(pklFile)

        leaderPosVelObss = leaderFollowerTimeRangeSumVelDict['leaderRangeSumVelObss']
        leaderSumValues = []
        for leaderPosVelObs in leaderPosVelObss:
            leaderSumValues.append(leaderPosVelObs[0])

        followerPosVelObss = leaderFollowerTimeRangeSumVelDict['followerRangeSumVelObss']
        followerSumValues = []
        for followerPosVelObs in followerPosVelObss:
            followerSumValues.append(followerPosVelObs[0])

        # Scale the plot
        f = plt.figure()
        f.set_figwidth(20)
        f.set_figheight(5)
        # Label
        oneCycleLength = 5000
        plt.xlabel('Timestep')
        plt.ylabel('Leader and follower sum of allRanges')
        plt.plot(range(0, oneCycleLength)
                 , leaderSumValues[0:oneCycleLength]
                 , label="Leader"
                 , color='red'
                 , linewidth=1)

        plt.plot(range(0, oneCycleLength)
                 , followerSumValues[0:oneCycleLength]
                 , label="follower"
                 , color='green'
                 , linewidth=1)
        # To show xlabel
        plt.tight_layout()

        # To show the inner labels
        plt.legend("Leader sum of allRanges, normal")

        # Novelty signal
        plt.show()



if __name__== "__main__":
    PlotSumTime.plotNormalLeaderSumTime()
    PlotSumTime.plotFollowLeaderSumTime()