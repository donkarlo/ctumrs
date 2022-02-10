from mUtility.database.file.RowsStartWithColNames import RowsStartWithColNames
from mDynamicSystem.obs.threeDPosVel.PosToVelObsSerieBuilderFromRowsStartWithColNames import \
    PosToVelObsSerieBuilderFromRowsStartWithColNames

dumpedTextFile = RowsStartWithColNames(
    "/home/donkarlo/Dropbox/projs/research/data/self-aware-drones/ctumrs/two-drones/inner-squares-follow-4/uav-2-gps.txt")

threeDMaker = PosToVelObsSerieBuilderFromRowsStartWithColNames(dumpedTextFile,
                                                               ["field.pose.pose.position.x"
                                                                            , "field.pose.pose.position.y"
                                                                            , "field.pose.pose.position.z"],
                                                               100000)
threeDMaker.saveToFileWithTime(
    "/home/donkarlo/Dropbox/projs/research/data/self-aware-drones/ctumrs/two-drones/inner-squares-follow-4/uav-2-cleaned-gps-pos-vel.txt"
    , ",")