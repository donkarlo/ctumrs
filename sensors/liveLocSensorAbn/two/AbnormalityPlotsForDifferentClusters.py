import matplotlib

import numpy as np
from matplotlib import pyplot as plt
import PyQt5

from ctumrs.sensors.liveLocSensorAbn.two.Abnormality import Abnormality


class AbnormalityPlots:
    def __init__(self):
        pass

    @staticmethod
    def plotGPSTimeAbnVals(abnormalitiesValues:list,styles:list):
        pass

if __name__=="__main__":
    matplotlib.use("Qt5Agg")
    plt.gca().set_aspect('equal')

    gpsSharedPathToTestScenarioAbnVals = "/home/donkarlo/Dropbox/projs/research/data/self-aware-drones/ctumrs/two-drones/follow-scenario/gps_origin/normal-scenario-trained/abnormality-values/"
    # gpsTimeAbnVals1 = np.asarray(Abnormality.loadTimeAbnVals(gpsSharedPath + "gaussianNoiseVarCo_0.32_training_360000_velco_1.05_clusters_75.pkl"))
    # gpsTimeAbnVals2 = np.asarray(Abnormality.loadTimeAbnVals(gpsSharedPath + "gaussianNoiseVarCo_0.16_training_360000_velco_1.05_clusters_75.pkl"))
    # gpsTimeAbnVals3 = np.asarray(Abnormality.loadTimeAbnVals(gpsSharedPath + "gaussianNoiseVarCo_0.08_training_360000_velco_1.05_clusters_75.pkl"))
    # gpsTimeAbnVals4 = np.asarray(Abnormality.loadTimeAbnVals(gpsSharedPath + "gaussianNoiseVarCo_0.04_training_360000_velco_1.05_clusters_75.pkl"))
    gpsTimeAbnVals5 = np.asarray(Abnormality.loadTimeAbnVals(gpsSharedPathToTestScenarioAbnVals + "gaussianNoiseVarCo_0_training_360000_velco_1.05_clusters_4.pkl"))
    gpsTimeAbnVals6 = np.asarray(Abnormality.loadTimeAbnVals(gpsSharedPathToTestScenarioAbnVals + "gaussianNoiseVarCo_0_training_360000_velco_1.05_clusters_8.pkl"))
    gpsTimeAbnVals7 = np.asarray(Abnormality.loadTimeAbnVals(gpsSharedPathToTestScenarioAbnVals + "gaussianNoiseVarCo_0_training_360000_velco_1.05_clusters_16.pkl"))
    gpsTimeAbnVals8 = np.asarray(Abnormality.loadTimeAbnVals(gpsSharedPathToTestScenarioAbnVals + "gaussianNoiseVarCo_0_training_360000_velco_1.05_clusters_24.pkl"))
    gpsTimeAbnVals9 = np.asarray(Abnormality.loadTimeAbnVals(gpsSharedPathToTestScenarioAbnVals + "gaussianNoiseVarCo_0_training_360000_velco_1.05_clusters_32.pkl"))
    gpsTimeAbnVals10 = np.asarray(Abnormality.loadTimeAbnVals(gpsSharedPathToTestScenarioAbnVals + "gaussianNoiseVarCo_0_training_360000_velco_1.05_clusters_48.pkl"))
    gpsTimeAbnVals11 = np.asarray(Abnormality.loadTimeAbnVals(gpsSharedPathToTestScenarioAbnVals + "gaussianNoiseVarCo_0_training_360000_velco_1.05_clusters_64.pkl"))
    gpsTimeAbnVals12 = np.asarray(Abnormality.loadTimeAbnVals(gpsSharedPathToTestScenarioAbnVals + "gaussianNoiseVarCo_0_training_360000_velco_1.05_clusters_75.pkl"))
    # plt.plot(gpsTimeAbnVals1[:,0],gpsTimeAbnVals1[:,1], label='0.32')
    # plt.plot(gpsTimeAbnVals2[:,0],gpsTimeAbnVals2[:,1], label='0.16')
    # plt.plot(gpsTimeAbnVals3[:,0],gpsTimeAbnVals3[:,1], label='0.08')
    # plt.plot(gpsTimeAbnVals4[:,0],gpsTimeAbnVals4[:,1], label='0.04')
    plt.plot(gpsTimeAbnVals5[:,0],gpsTimeAbnVals5[:,1], label='4')
    plt.plot(gpsTimeAbnVals6[:,0],gpsTimeAbnVals6[:,1], label='8')
    plt.plot(gpsTimeAbnVals7[:,0],gpsTimeAbnVals7[:,1], label='16')
    plt.plot(gpsTimeAbnVals8[:,0],gpsTimeAbnVals8[:,1], label='24')
    plt.plot(gpsTimeAbnVals9[:,0],gpsTimeAbnVals9[:,1], label='32')
    plt.plot(gpsTimeAbnVals10[:,0],gpsTimeAbnVals10[:,1], label='48')
    plt.plot(gpsTimeAbnVals11[:,0],gpsTimeAbnVals11[:,1], label='64')
    plt.plot(gpsTimeAbnVals12[:,0],gpsTimeAbnVals12[:,1], label='75')
    plt.legend()
    plt.plot()
    plt.show()

    lidarSharedPathToTestScenarioAbnVals = "/home/donkarlo/Dropbox/projs/research/data/self-aware-drones/ctumrs/two-drones/follow-scenario/rplidar/normal-scenario-trained/abnormality-values/"
    # lidarTimeAbnVals1 = np.asarray(Abnormality.loadTimeAbnVals(lidarSharedPath + "gaussianNoiseVarCo_0.32_training_120000_velco_0.5_clusters_32_autoencoder_latentdim_3_epochs_200.pkl"))
    # lidarTimeAbnVals2 = np.asarray(Abnormality.loadTimeAbnVals(lidarSharedPath + "gaussianNoiseVarCo_0.16_training_120000_velco_0.5_clusters_32_autoencoder_latentdim_3_epochs_200.pkl"))
    # lidarTimeAbnVals3 = np.asarray(Abnormality.loadTimeAbnVals(lidarSharedPath + "gaussianNoiseVarCo_0.08_training_120000_velco_0.5_clusters_32_autoencoder_latentdim_3_epochs_200.pkl"))
    # lidarTimeAbnVals4 = np.asarray(Abnormality.loadTimeAbnVals(lidarSharedPath + "gaussianNoiseVarCo_0.04_training_120000_velco_0.5_clusters_32_autoencoder_latentdim_3_epochs_200.pkl"))
    lidarTimeAbnVals5 = np.asarray(Abnormality.loadTimeAbnVals(lidarSharedPathToTestScenarioAbnVals + "gaussianNoiseVarCo_0_training_120000_velco_0.5_clusters_4_autoencoder_latentdim_3_epochs_200.pkl"))
    lidarTimeAbnVals6 = np.asarray(Abnormality.loadTimeAbnVals(lidarSharedPathToTestScenarioAbnVals + "gaussianNoiseVarCo_0_training_120000_velco_0.5_clusters_8_autoencoder_latentdim_3_epochs_200.pkl"))
    lidarTimeAbnVals7 = np.asarray(Abnormality.loadTimeAbnVals(lidarSharedPathToTestScenarioAbnVals + "gaussianNoiseVarCo_0_training_120000_velco_0.5_clusters_16_autoencoder_latentdim_3_epochs_200.pkl"))
    lidarTimeAbnVals8 = np.asarray(Abnormality.loadTimeAbnVals(lidarSharedPathToTestScenarioAbnVals + "gaussianNoiseVarCo_0_training_120000_velco_0.5_clusters_24_autoencoder_latentdim_3_epochs_200.pkl"))
    lidarTimeAbnVals9 = np.asarray(Abnormality.loadTimeAbnVals(lidarSharedPathToTestScenarioAbnVals + "gaussianNoiseVarCo_0_training_120000_velco_0.5_clusters_32_autoencoder_latentdim_3_epochs_200.pkl"))
    lidarTimeAbnVals10 = np.asarray(Abnormality.loadTimeAbnVals(lidarSharedPathToTestScenarioAbnVals + "gaussianNoiseVarCo_0_training_120000_velco_0.5_clusters_48_autoencoder_latentdim_3_epochs_200.pkl"))
    lidarTimeAbnVals11 = np.asarray(Abnormality.loadTimeAbnVals(lidarSharedPathToTestScenarioAbnVals + "gaussianNoiseVarCo_0_training_120000_velco_0.5_clusters_64_autoencoder_latentdim_3_epochs_200.pkl"))
    lidarTimeAbnVals12 = np.asarray(Abnormality.loadTimeAbnVals(lidarSharedPathToTestScenarioAbnVals + "gaussianNoiseVarCo_0_training_120000_velco_0.5_clusters_75_autoencoder_latentdim_3_epochs_200.pkl"))
    # plt.plot(lidarTimeAbnVals1[:, 0], lidarTimeAbnVals1[:, 1], label='0.32')
    # plt.plot(lidarTimeAbnVals2[:, 0], lidarTimeAbnVals2[:, 1], label='0.16')
    # plt.plot(lidarTimeAbnVals3[:, 0], lidarTimeAbnVals3[:, 1], label='0.08')
    # plt.plot(lidarTimeAbnVals4[:, 0], lidarTimeAbnVals4[:, 1], label='0.04')
    plt.plot(lidarTimeAbnVals5[:, 0], lidarTimeAbnVals5[:, 1], label='4')
    plt.plot(lidarTimeAbnVals6[:, 0], lidarTimeAbnVals6[:, 1], label='8')
    plt.plot(lidarTimeAbnVals7[:, 0], lidarTimeAbnVals7[:, 1], label='16')
    plt.plot(lidarTimeAbnVals8[:, 0], lidarTimeAbnVals8[:, 1], label='24')
    plt.plot(lidarTimeAbnVals9[:, 0], lidarTimeAbnVals9[:, 1], label='32')
    plt.plot(lidarTimeAbnVals10[:, 0], lidarTimeAbnVals10[:, 1], label='48')
    plt.plot(lidarTimeAbnVals11[:, 0], lidarTimeAbnVals11[:, 1], label='64')
    plt.plot(lidarTimeAbnVals12[:, 0], lidarTimeAbnVals12[:, 1], label='75')
    plt.legend()
    plt.plot()
    plt.show()





