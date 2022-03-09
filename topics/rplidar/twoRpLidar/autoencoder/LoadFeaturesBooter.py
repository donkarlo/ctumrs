from keras.models import load_model
class LoadFeaturesBooter():
    @staticmethod
    def loadEncoder():
        trainedAutoencoder = load_model('savedEncoder.h5')
        trainedAutoencoder.summary()
        trainedAutoencoder.predict()


if __name__=="__main__":
    LoadFeaturesBooter.loadEncoder()