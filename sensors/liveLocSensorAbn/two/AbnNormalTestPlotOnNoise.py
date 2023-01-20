import numpy as np

from ctumrs.sensors.liveLocSensorAbn.two.Abnormality import Abnormality

if __name__ == "__main__":
    sharedPathUntilScenarios = "/home/donkarlo/Dropbox/projs/research/data/self-aware-drones/ctumrs/two-drones/"

    #GPS
    gpsSharedPathToNormalScenarioAbnVals = sharedPathUntilScenarios + "normal-scenario/gps_origin/normal-scenario-trained/abnormality-values/"
    gpsSharedPathToFollowScenarioAbnVals = sharedPathUntilScenarios + "follow-scenario/gps_origin/normal-scenario-trained/abnormality-values/"

    gpsNormalTimeAbnValsNoise0 = np.asarray(Abnormality.loadTimeAbnVals(gpsSharedPathToNormalScenarioAbnVals + "gaussianNoiseVarCo_0_training_360000_velco_1.05_clusters_75.pkl"))
    gpsFollowTimeAbnValsNoise0 = np.asarray(Abnormality.loadTimeAbnVals(gpsSharedPathToFollowScenarioAbnVals + "gaussianNoiseVarCo_0_training_360000_velco_1.05_clusters_75.pkl"))

    gpsNormalTimeAbnValsNoise0_00125 = np.asarray(Abnormality.loadTimeAbnVals(gpsSharedPathToNormalScenarioAbnVals + "gaussianNoiseVarCo_0.00125_training_360000_velco_1.05_clusters_75.pkl"))
    gpsFollowTimeAbnValsNoise0_00125 = np.asarray(Abnormality.loadTimeAbnVals(gpsSharedPathToFollowScenarioAbnVals + "gaussianNoiseVarCo_0.00125_training_360000_velco_1.05_clusters_75.pkl"))

    gpsNormalTimeAbnValsNoise0_0025 = np.asarray(Abnormality.loadTimeAbnVals(gpsSharedPathToNormalScenarioAbnVals + "gaussianNoiseVarCo_0.0025_training_360000_velco_1.05_clusters_75.pkl"))
    gpsFollowTimeAbnValsNoise0_0025 = np.asarray(Abnormality.loadTimeAbnVals(gpsSharedPathToFollowScenarioAbnVals + "gaussianNoiseVarCo_0.0025_training_360000_velco_1.05_clusters_75.pkl"))

    gpsNormalTimeAbnValsNoise0_005 = np.asarray(Abnormality.loadTimeAbnVals(gpsSharedPathToNormalScenarioAbnVals + "gaussianNoiseVarCo_0.005_training_360000_velco_1.05_clusters_75.pkl"))
    gpsFollowTimeAbnValsNoise0_005 = np.asarray(Abnormality.loadTimeAbnVals(gpsSharedPathToFollowScenarioAbnVals + "gaussianNoiseVarCo_0.005_training_360000_velco_1.05_clusters_75.pkl"))

    gpsNormalTimeAbnValsNoise0_01 = np.asarray(Abnormality.loadTimeAbnVals(gpsSharedPathToNormalScenarioAbnVals + "gaussianNoiseVarCo_0.01_training_360000_velco_1.05_clusters_75.pkl"))
    gpsFollowTimeAbnValsNoise0_01 = np.asarray(Abnormality.loadTimeAbnVals(gpsSharedPathToFollowScenarioAbnVals + "gaussianNoiseVarCo_0.01_training_360000_velco_1.05_clusters_75.pkl"))

    gpsNormalTimeAbnValsNoise0_02 = np.asarray(Abnormality.loadTimeAbnVals(gpsSharedPathToNormalScenarioAbnVals + "gaussianNoiseVarCo_0.02_training_360000_velco_1.05_clusters_75.pkl"))
    gpsFollowTimeAbnValsNoise0_02 = np.asarray(Abnormality.loadTimeAbnVals(gpsSharedPathToFollowScenarioAbnVals + "gaussianNoiseVarCo_0.02_training_360000_velco_1.05_clusters_75.pkl"))

    gpsNormalTimeAbnValsNoise0_04 = np.asarray(Abnormality.loadTimeAbnVals(gpsSharedPathToNormalScenarioAbnVals + "gaussianNoiseVarCo_0.04_training_360000_velco_1.05_clusters_75.pkl"))
    gpsFollowTimeAbnValsNoise0_04 = np.asarray(Abnormality.loadTimeAbnVals(gpsSharedPathToFollowScenarioAbnVals + "gaussianNoiseVarCo_0.04_training_360000_velco_1.05_clusters_75.pkl"))

    gpsNormalTimeAbnValsNoise0_08 = np.asarray(Abnormality.loadTimeAbnVals(gpsSharedPathToNormalScenarioAbnVals + "gaussianNoiseVarCo_0.08_training_360000_velco_1.05_clusters_75.pkl"))
    gpsFollowTimeAbnValsNoise0_08 = np.asarray(Abnormality.loadTimeAbnVals(gpsSharedPathToFollowScenarioAbnVals + "gaussianNoiseVarCo_0.08_training_360000_velco_1.05_clusters_75.pkl"))

    gpsNormalTimeAbnValsNoise0_16 = np.asarray(Abnormality.loadTimeAbnVals(gpsSharedPathToNormalScenarioAbnVals + "gaussianNoiseVarCo_0.16_training_360000_velco_1.05_clusters_75.pkl"))
    gpsFollowTimeAbnValsNoise0_16 = np.asarray(Abnormality.loadTimeAbnVals(gpsSharedPathToFollowScenarioAbnVals + "gaussianNoiseVarCo_0.16_training_360000_velco_1.05_clusters_75.pkl"))

    gpsNormalTimeAbnValsNoise0_32 = np.asarray(Abnormality.loadTimeAbnVals(gpsSharedPathToNormalScenarioAbnVals + "gaussianNoiseVarCo_0.32_training_360000_velco_1.05_clusters_75.pkl"))
    gpsFollowTimeAbnValsNoise0_32 = np.asarray(Abnormality.loadTimeAbnVals(gpsSharedPathToFollowScenarioAbnVals + "gaussianNoiseVarCo_0.32_training_360000_velco_1.05_clusters_75.pkl"))

    #LIDAR
    lidarSharedPathToNormalScenarioAbnVals = sharedPathUntilScenarios + "normal-scenario/rplidar/normal-scenario-trained/abnormality-values/"
    lidarSharedPathToFollowScenarioAbnVals = sharedPathUntilScenarios + "follow-scenario/rplidar/normal-scenario-trained/abnormality-values/"

    lidarNormalTimeAbnValsNoise0 = np.asarray(Abnormality.loadTimeAbnVals(lidarSharedPathToNormalScenarioAbnVals + "gaussianNoiseVarCo_0_training_120000_velco_0.5_clusters_32_autoencoder_latentdim_3_epochs_200.pkl"))
    lidarFollowTimeAbnValsNoise0 = np.asarray(Abnormality.loadTimeAbnVals(lidarSharedPathToFollowScenarioAbnVals + "gaussianNoiseVarCo_0_training_120000_velco_0.5_clusters_32_autoencoder_latentdim_3_epochs_200.pkl"))

    lidarNormalTimeAbnValsNoise0_00125 = np.asarray(Abnormality.loadTimeAbnVals(lidarSharedPathToNormalScenarioAbnVals + "gaussianNoiseVarCo_0.00125_training_120000_velco_0.5_clusters_32_autoencoder_latentdim_3_epochs_200.pkl"))
    lidarFollowTimeAbnValsNoise0_00125 = np.asarray(Abnormality.loadTimeAbnVals(lidarSharedPathToFollowScenarioAbnVals + "gaussianNoiseVarCo_0.00125_training_120000_velco_0.5_clusters_32_autoencoder_latentdim_3_epochs_200.pkl"))

    lidarNormalTimeAbnValsNoise0_0025 = np.asarray(Abnormality.loadTimeAbnVals(lidarSharedPathToNormalScenarioAbnVals + "gaussianNoiseVarCo_0.0025_training_120000_velco_0.5_clusters_32_autoencoder_latentdim_3_epochs_200.pkl"))
    lidarFollowTimeAbnValsNoise0_0025 = np.asarray(Abnormality.loadTimeAbnVals(lidarSharedPathToFollowScenarioAbnVals + "gaussianNoiseVarCo_0.0025_training_120000_velco_0.5_clusters_32_autoencoder_latentdim_3_epochs_200.pkl"))

    lidarNormalTimeAbnValsNoise0_005 = np.asarray(Abnormality.loadTimeAbnVals(lidarSharedPathToNormalScenarioAbnVals + "gaussianNoiseVarCo_0.005_training_120000_velco_0.5_clusters_32_autoencoder_latentdim_3_epochs_200.pkl"))
    lidarFollowTimeAbnValsNoise0_005 = np.asarray(Abnormality.loadTimeAbnVals(lidarSharedPathToFollowScenarioAbnVals + "gaussianNoiseVarCo_0.005_training_120000_velco_0.5_clusters_32_autoencoder_latentdim_3_epochs_200.pkl"))

    lidarNormalTimeAbnValsNoise0_01 = np.asarray(Abnormality.loadTimeAbnVals(lidarSharedPathToNormalScenarioAbnVals + "gaussianNoiseVarCo_0.01_training_120000_velco_0.5_clusters_32_autoencoder_latentdim_3_epochs_200.pkl"))
    lidarFollowTimeAbnValsNoise0_01 = np.asarray(Abnormality.loadTimeAbnVals(lidarSharedPathToFollowScenarioAbnVals + "gaussianNoiseVarCo_0.01_training_120000_velco_0.5_clusters_32_autoencoder_latentdim_3_epochs_200.pkl"))

    lidarNormalTimeAbnValsNoise0_02 = np.asarray(Abnormality.loadTimeAbnVals(lidarSharedPathToNormalScenarioAbnVals + "gaussianNoiseVarCo_0.02_training_120000_velco_0.5_clusters_32_autoencoder_latentdim_3_epochs_200.pkl"))
    lidarFollowTimeAbnValsNoise0_02 = np.asarray(Abnormality.loadTimeAbnVals(lidarSharedPathToFollowScenarioAbnVals + "gaussianNoiseVarCo_0.02_training_120000_velco_0.5_clusters_32_autoencoder_latentdim_3_epochs_200.pkl"))

    lidarNormalTimeAbnValsNoise0_04 = np.asarray(Abnormality.loadTimeAbnVals(lidarSharedPathToNormalScenarioAbnVals + "gaussianNoiseVarCo_0.04_training_120000_velco_0.5_clusters_32_autoencoder_latentdim_3_epochs_200.pkl"))
    lidarFollowTimeAbnValsNoise0_04 = np.asarray(Abnormality.loadTimeAbnVals(lidarSharedPathToFollowScenarioAbnVals + "gaussianNoiseVarCo_0.04_training_120000_velco_0.5_clusters_32_autoencoder_latentdim_3_epochs_200.pkl"))

    lidarNormalTimeAbnValsNoise0_08 = np.asarray(Abnormality.loadTimeAbnVals(lidarSharedPathToNormalScenarioAbnVals + "gaussianNoiseVarCo_0.08_training_120000_velco_0.5_clusters_32_autoencoder_latentdim_3_epochs_200.pkl"))
    lidarFollowTimeAbnValsNoise0_08 = np.asarray(Abnormality.loadTimeAbnVals(lidarSharedPathToFollowScenarioAbnVals + "gaussianNoiseVarCo_0.08_training_120000_velco_0.5_clusters_32_autoencoder_latentdim_3_epochs_200.pkl"))

    lidarNormalTimeAbnValsNoise0_16 = np.asarray(Abnormality.loadTimeAbnVals(lidarSharedPathToNormalScenarioAbnVals + "gaussianNoiseVarCo_0.16_training_120000_velco_0.5_clusters_32_autoencoder_latentdim_3_epochs_200.pkl"))
    lidarFollowTimeAbnValsNoise0_16 = np.asarray(Abnormality.loadTimeAbnVals(lidarSharedPathToFollowScenarioAbnVals + "gaussianNoiseVarCo_0.16_training_120000_velco_0.5_clusters_32_autoencoder_latentdim_3_epochs_200.pkl"))

    lidarNormalTimeAbnValsNoise0_32 = np.asarray(Abnormality.loadTimeAbnVals(lidarSharedPathToNormalScenarioAbnVals + "gaussianNoiseVarCo_0.32_training_120000_velco_0.5_clusters_32_autoencoder_latentdim_3_epochs_200.pkl"))
    lidarFollowTimeAbnValsNoise0_32 = np.asarray(Abnormality.loadTimeAbnVals(lidarSharedPathToFollowScenarioAbnVals + "gaussianNoiseVarCo_0.32_training_120000_velco_0.5_clusters_32_autoencoder_latentdim_3_epochs_200.pkl"))

    Abnormality.plotNew(
        [
              (gpsNormalTimeAbnValsNoise0, gpsFollowTimeAbnValsNoise0, "noise=0.0002")
            , (gpsNormalTimeAbnValsNoise0_00125, gpsFollowTimeAbnValsNoise0_00125, "noise=0.00125")
            , (gpsNormalTimeAbnValsNoise0_0025, gpsFollowTimeAbnValsNoise0_0025, "noise=0.0025")
            , (gpsNormalTimeAbnValsNoise0_005, gpsFollowTimeAbnValsNoise0_005, "noise=0.005")
            # , (gpsNormalTimeAbnValsNoise0_01, gpsFollowTimeAbnValsNoise0_01, "noise=0.01")
            # , (gpsNormalTimeAbnValsNoise0_02, gpsFollowTimeAbnValsNoise0_02, "noise=0.02")
            # , (gpsNormalTimeAbnValsNoise0_04, gpsFollowTimeAbnValsNoise0_04, "noise=0.04")
            # , (gpsNormalTimeAbnValsNoise0_08, gpsFollowTimeAbnValsNoise0_08, "noise=0.08")
            # , (gpsNormalTimeAbnValsNoise0_16, gpsFollowTimeAbnValsNoise0_16, "noise=0.16")
            # , (gpsNormalTimeAbnValsNoise0_32, gpsFollowTimeAbnValsNoise0_32, "noise=0.32")
        ]
        ,
        [
              (lidarNormalTimeAbnValsNoise0, lidarFollowTimeAbnValsNoise0, "noise=0.0002")
            , (lidarNormalTimeAbnValsNoise0_00125, lidarFollowTimeAbnValsNoise0_00125, "noise=0.00125")
            , (lidarNormalTimeAbnValsNoise0_0025, lidarFollowTimeAbnValsNoise0_0025, "noise=0.0025")
            , (lidarNormalTimeAbnValsNoise0_005, lidarFollowTimeAbnValsNoise0_005, "noise=0.005")
            # , (lidarNormalTimeAbnValsNoise0_01, lidarFollowTimeAbnValsNoise0_01, "noise=0.01")
            # , (lidarNormalTimeAbnValsNoise0_02, lidarFollowTimeAbnValsNoise0_02, "noise=0.02")
            # , (lidarNormalTimeAbnValsNoise0_04, lidarFollowTimeAbnValsNoise0_04, "noise=0.04")
            # , (lidarNormalTimeAbnValsNoise0_08, lidarFollowTimeAbnValsNoise0_08, "noise=0.08")
            # , (lidarNormalTimeAbnValsNoise0_16, lidarFollowTimeAbnValsNoise0_16, "noise=0.16")
            # , (lidarNormalTimeAbnValsNoise0_32, lidarFollowTimeAbnValsNoise0_32, "noise=0.32")
         ]
    )