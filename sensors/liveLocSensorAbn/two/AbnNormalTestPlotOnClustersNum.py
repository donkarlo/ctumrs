import numpy as np

from ctumrs.sensors.liveLocSensorAbn.two.Abnormality import Abnormality

if __name__ == "__main__":
    sharedPathUntilScenarios = "/home/donkarlo/Dropbox/projs/research/data/self-aware-drones/ctumrs/two-drones/"

    ## GPS
    gpsSharedPathToNormalScenarioAbnVals = sharedPathUntilScenarios + "normal-scenario/gps_origin/normal-scenario-trained/abnormality-values/"
    gpsSharedPathToFollowScenarioAbnVals = sharedPathUntilScenarios + "follow-scenario/gps_origin/normal-scenario-trained/abnormality-values/"

    gpsNormalTimeAbnValsCLusters100 = np.asarray(Abnormality.loadTimeAbnVals(gpsSharedPathToNormalScenarioAbnVals + "gaussianNoiseVarCo_0_training_360000_velco_1.05_clusters_100.pkl"))
    gpsFollowTimeAbnValsClusters100 = np.asarray(Abnormality.loadTimeAbnVals(gpsSharedPathToFollowScenarioAbnVals + "gaussianNoiseVarCo_0_training_360000_velco_1.05_clusters_100.pkl"))

    gpsNormalTimeAbnValsCLusters75 = np.asarray(Abnormality.loadTimeAbnVals(gpsSharedPathToNormalScenarioAbnVals + "gaussianNoiseVarCo_0_training_360000_velco_1.05_clusters_75.pkl"))
    gpsFollowTimeAbnValsClusters75 = np.asarray(Abnormality.loadTimeAbnVals(gpsSharedPathToFollowScenarioAbnVals + "gaussianNoiseVarCo_0_training_360000_velco_1.05_clusters_75.pkl"))


    gpsNormalTimeAbnValsCLusters64 = np.asarray(Abnormality.loadTimeAbnVals(gpsSharedPathToNormalScenarioAbnVals + "gaussianNoiseVarCo_0_training_360000_velco_1.05_clusters_64.pkl"))
    gpsFollowTimeAbnValsClusters64 = np.asarray(Abnormality.loadTimeAbnVals(gpsSharedPathToFollowScenarioAbnVals + "gaussianNoiseVarCo_0_training_360000_velco_1.05_clusters_64.pkl"))

    gpsNormalTimeAbnValsCLusters48 = np.asarray(Abnormality.loadTimeAbnVals(gpsSharedPathToNormalScenarioAbnVals + "gaussianNoiseVarCo_0_training_360000_velco_1.05_clusters_48.pkl"))
    gpsFollowTimeAbnValsClusters48 = np.asarray(Abnormality.loadTimeAbnVals(gpsSharedPathToFollowScenarioAbnVals + "gaussianNoiseVarCo_0_training_360000_velco_1.05_clusters_48.pkl"))

    gpsNormalTimeAbnValsCLusters40 = np.asarray(Abnormality.loadTimeAbnVals(gpsSharedPathToNormalScenarioAbnVals + "gaussianNoiseVarCo_0_training_360000_velco_1.05_clusters_40.pkl"))
    gpsFollowTimeAbnValsClusters40 = np.asarray(Abnormality.loadTimeAbnVals(gpsSharedPathToFollowScenarioAbnVals + "gaussianNoiseVarCo_0_training_360000_velco_1.05_clusters_40.pkl"))

    gpsNormalTimeAbnValsCLusters32 = np.asarray(Abnormality.loadTimeAbnVals(gpsSharedPathToNormalScenarioAbnVals + "gaussianNoiseVarCo_0_training_360000_velco_1.05_clusters_32.pkl"))
    gpsFollowTimeAbnValsClusters32 = np.asarray(Abnormality.loadTimeAbnVals(gpsSharedPathToFollowScenarioAbnVals + "gaussianNoiseVarCo_0_training_360000_velco_1.05_clusters_32.pkl"))

    gpsNormalTimeAbnValsCLusters24 = np.asarray(Abnormality.loadTimeAbnVals(gpsSharedPathToNormalScenarioAbnVals + "gaussianNoiseVarCo_0_training_360000_velco_1.05_clusters_24.pkl"))
    gpsFollowTimeAbnValsClusters24 = np.asarray(Abnormality.loadTimeAbnVals(gpsSharedPathToFollowScenarioAbnVals + "gaussianNoiseVarCo_0_training_360000_velco_1.05_clusters_24.pkl"))

    gpsNormalTimeAbnValsCLusters16 = np.asarray(Abnormality.loadTimeAbnVals(gpsSharedPathToNormalScenarioAbnVals + "gaussianNoiseVarCo_0_training_360000_velco_1.05_clusters_16.pkl"))
    gpsFollowTimeAbnValsClusters16 = np.asarray(Abnormality.loadTimeAbnVals(gpsSharedPathToFollowScenarioAbnVals + "gaussianNoiseVarCo_0_training_360000_velco_1.05_clusters_16.pkl"))

    gpsNormalTimeAbnValsCLusters12 = np.asarray(Abnormality.loadTimeAbnVals(gpsSharedPathToNormalScenarioAbnVals + "gaussianNoiseVarCo_0_training_360000_velco_1.05_clusters_12.pkl"))
    gpsFollowTimeAbnValsClusters12 = np.asarray(Abnormality.loadTimeAbnVals(gpsSharedPathToFollowScenarioAbnVals + "gaussianNoiseVarCo_0_training_360000_velco_1.05_clusters_12.pkl"))

    gpsNormalTimeAbnValsCLusters8 = np.asarray(Abnormality.loadTimeAbnVals(gpsSharedPathToNormalScenarioAbnVals + "gaussianNoiseVarCo_0_training_360000_velco_1.05_clusters_8.pkl"))
    gpsFollowTimeAbnValsClusters8 = np.asarray(Abnormality.loadTimeAbnVals(gpsSharedPathToFollowScenarioAbnVals + "gaussianNoiseVarCo_0_training_360000_velco_1.05_clusters_8.pkl"))

    gpsNormalTimeAbnValsCLusters4 = np.asarray(Abnormality.loadTimeAbnVals(gpsSharedPathToNormalScenarioAbnVals + "gaussianNoiseVarCo_0_training_360000_velco_1.05_clusters_4.pkl"))
    gpsFollowTimeAbnValsClusters4 = np.asarray(Abnormality.loadTimeAbnVals(gpsSharedPathToFollowScenarioAbnVals + "gaussianNoiseVarCo_0_training_360000_velco_1.05_clusters_4.pkl"))

    ## LIDAR
    lidarSharedPathToNormalScenarioAbnVals = sharedPathUntilScenarios + "normal-scenario/rplidar/normal-scenario-trained/abnormality-values/"
    lidarSharedPathToFollowScenarioAbnVals = sharedPathUntilScenarios + "follow-scenario/rplidar/normal-scenario-trained/abnormality-values/"

    lidarNormalTimeAbnValsClusters100 = np.asarray(Abnormality.loadTimeAbnVals(lidarSharedPathToNormalScenarioAbnVals + "gaussianNoiseVarCo_0_training_120000_velco_0.5_clusters_100_autoencoder_latentdim_3_epochs_200.pkl"))
    lidarFollowTimeAbnValsClusters100 = np.asarray(Abnormality.loadTimeAbnVals(lidarSharedPathToFollowScenarioAbnVals + "gaussianNoiseVarCo_0_training_120000_velco_0.5_clusters_100_autoencoder_latentdim_3_epochs_200.pkl"))

    lidarNormalTimeAbnValsClusters75 = np.asarray(Abnormality.loadTimeAbnVals(lidarSharedPathToNormalScenarioAbnVals + "gaussianNoiseVarCo_0_training_120000_velco_0.5_clusters_75_autoencoder_latentdim_3_epochs_200.pkl"))
    lidarFollowTimeAbnValsClusters75 = np.asarray(Abnormality.loadTimeAbnVals(lidarSharedPathToFollowScenarioAbnVals + "gaussianNoiseVarCo_0_training_120000_velco_0.5_clusters_75_autoencoder_latentdim_3_epochs_200.pkl"))

    lidarNormalTimeAbnValsClusters64 = np.asarray(Abnormality.loadTimeAbnVals(lidarSharedPathToNormalScenarioAbnVals + "gaussianNoiseVarCo_0_training_120000_velco_0.5_clusters_64_autoencoder_latentdim_3_epochs_200.pkl"))
    lidarFollowTimeAbnValsClusters64 = np.asarray(Abnormality.loadTimeAbnVals(lidarSharedPathToFollowScenarioAbnVals + "gaussianNoiseVarCo_0_training_120000_velco_0.5_clusters_64_autoencoder_latentdim_3_epochs_200.pkl"))

    lidarNormalTimeAbnValsClusters48 = np.asarray(Abnormality.loadTimeAbnVals(lidarSharedPathToNormalScenarioAbnVals + "gaussianNoiseVarCo_0_training_120000_velco_0.5_clusters_48_autoencoder_latentdim_3_epochs_200.pkl"))
    lidarFollowTimeAbnValsClusters48 = np.asarray(Abnormality.loadTimeAbnVals(lidarSharedPathToFollowScenarioAbnVals + "gaussianNoiseVarCo_0_training_120000_velco_0.5_clusters_48_autoencoder_latentdim_3_epochs_200.pkl"))

    lidarNormalTimeAbnValsClusters40 = np.asarray(Abnormality.loadTimeAbnVals(lidarSharedPathToNormalScenarioAbnVals + "gaussianNoiseVarCo_0_training_120000_velco_0.5_clusters_40_autoencoder_latentdim_3_epochs_200.pkl"))
    lidarFollowTimeAbnValsClusters40 = np.asarray(Abnormality.loadTimeAbnVals(lidarSharedPathToFollowScenarioAbnVals + "gaussianNoiseVarCo_0_training_120000_velco_0.5_clusters_40_autoencoder_latentdim_3_epochs_200.pkl"))

    lidarNormalTimeAbnValsClusters32 = np.asarray(Abnormality.loadTimeAbnVals(lidarSharedPathToNormalScenarioAbnVals + "gaussianNoiseVarCo_0_training_120000_velco_0.5_clusters_32_autoencoder_latentdim_3_epochs_200.pkl"))
    lidarFollowTimeAbnValsClusters32 = np.asarray(Abnormality.loadTimeAbnVals(lidarSharedPathToFollowScenarioAbnVals + "gaussianNoiseVarCo_0_training_120000_velco_0.5_clusters_32_autoencoder_latentdim_3_epochs_200.pkl"))

    lidarNormalTimeAbnValsClusters24 = np.asarray(Abnormality.loadTimeAbnVals(lidarSharedPathToNormalScenarioAbnVals + "gaussianNoiseVarCo_0_training_120000_velco_0.5_clusters_24_autoencoder_latentdim_3_epochs_200.pkl"))
    lidarFollowTimeAbnValsClusters24 = np.asarray(Abnormality.loadTimeAbnVals(lidarSharedPathToFollowScenarioAbnVals + "gaussianNoiseVarCo_0_training_120000_velco_0.5_clusters_24_autoencoder_latentdim_3_epochs_200.pkl"))

    lidarNormalTimeAbnValsClusters16 = np.asarray(Abnormality.loadTimeAbnVals(lidarSharedPathToNormalScenarioAbnVals + "gaussianNoiseVarCo_0_training_120000_velco_0.5_clusters_16_autoencoder_latentdim_3_epochs_200.pkl"))
    lidarFollowTimeAbnValsClusters16 = np.asarray(Abnormality.loadTimeAbnVals(lidarSharedPathToFollowScenarioAbnVals + "gaussianNoiseVarCo_0_training_120000_velco_0.5_clusters_16_autoencoder_latentdim_3_epochs_200.pkl"))

    lidarNormalTimeAbnValsClusters12 = np.asarray(Abnormality.loadTimeAbnVals(lidarSharedPathToNormalScenarioAbnVals + "gaussianNoiseVarCo_0_training_120000_velco_0.5_clusters_12_autoencoder_latentdim_3_epochs_200.pkl"))
    lidarFollowTimeAbnValsClusters12 = np.asarray(Abnormality.loadTimeAbnVals(lidarSharedPathToFollowScenarioAbnVals + "gaussianNoiseVarCo_0_training_120000_velco_0.5_clusters_12_autoencoder_latentdim_3_epochs_200.pkl"))

    lidarNormalTimeAbnValsClusters8 = np.asarray(Abnormality.loadTimeAbnVals(lidarSharedPathToNormalScenarioAbnVals + "gaussianNoiseVarCo_0_training_120000_velco_0.5_clusters_8_autoencoder_latentdim_3_epochs_200.pkl"))
    lidarFollowTimeAbnValsClusters8 = np.asarray(Abnormality.loadTimeAbnVals(lidarSharedPathToFollowScenarioAbnVals + "gaussianNoiseVarCo_0_training_120000_velco_0.5_clusters_8_autoencoder_latentdim_3_epochs_200.pkl"))

    lidarNormalTimeAbnValsClusters4 = np.asarray(Abnormality.loadTimeAbnVals(lidarSharedPathToNormalScenarioAbnVals + "gaussianNoiseVarCo_0_training_120000_velco_0.5_clusters_4_autoencoder_latentdim_3_epochs_200.pkl"))
    lidarFollowTimeAbnValsClusters4 = np.asarray(Abnormality.loadTimeAbnVals(lidarSharedPathToFollowScenarioAbnVals + "gaussianNoiseVarCo_0_training_120000_velco_0.5_clusters_4_autoencoder_latentdim_3_epochs_200.pkl"))

    Abnormality.plotAbnValsTolerenceLines\
    (
        [
            # (gpsNormalTimeAbnValsCLusters100,gpsFollowTimeAbnValsClusters100,"clusters=100"),
            (gpsNormalTimeAbnValsCLusters75,gpsFollowTimeAbnValsClusters75,"clusters=75"),
            # (gpsNormalTimeAbnValsCLusters64,gpsFollowTimeAbnValsClusters64,"clusters=64"),
            # (gpsNormalTimeAbnValsCLusters48,gpsFollowTimeAbnValsClusters48,"clusters=48"),
            # (gpsNormalTimeAbnValsCLusters40,gpsFollowTimeAbnValsClusters40,"clusters=40"),
            (gpsNormalTimeAbnValsCLusters32,gpsFollowTimeAbnValsClusters32,"clusters=32"),
            (gpsNormalTimeAbnValsCLusters24,gpsFollowTimeAbnValsClusters24,"clusters=24"),
            # (gpsNormalTimeAbnValsCLusters16,gpsFollowTimeAbnValsClusters16,"clusters=16"),
            (gpsNormalTimeAbnValsCLusters12,gpsFollowTimeAbnValsClusters12,"clusters=12"),
            # (gpsNormalTimeAbnValsCLusters8,gpsFollowTimeAbnValsClusters8,"clusters=8"),
            # (gpsNormalTimeAbnValsCLusters4,gpsFollowTimeAbnValsClusters4,"clusters=4"),
        ]
        ,
        [
            # (lidarNormalTimeAbnValsClusters100, lidarFollowTimeAbnValsClusters100, "clusters=100"),
            (lidarNormalTimeAbnValsClusters75, lidarFollowTimeAbnValsClusters75, "clusters=75"),
            # (lidarNormalTimeAbnValsClusters64, lidarFollowTimeAbnValsClusters64, "clusters=64"),
            # (lidarNormalTimeAbnValsClusters48, lidarFollowTimeAbnValsClusters48, "clusters=48"),
            # (lidarNormalTimeAbnValsClusters40, lidarFollowTimeAbnValsClusters40, "clusters=40"),
            (lidarNormalTimeAbnValsClusters32, lidarFollowTimeAbnValsClusters32, "clusters=32"),
            (lidarNormalTimeAbnValsClusters24, lidarFollowTimeAbnValsClusters24, "clusters=24"),
            # (lidarNormalTimeAbnValsClusters16, lidarFollowTimeAbnValsClusters16, "clusters=16"),
            (lidarNormalTimeAbnValsClusters12, lidarFollowTimeAbnValsClusters12, "clusters=12"),
            # (lidarNormalTimeAbnValsClusters8, lidarFollowTimeAbnValsClusters8, "clusters=8"),
            # (lidarNormalTimeAbnValsClusters4, lidarFollowTimeAbnValsClusters4, "clusters=4"),
         ]
    )