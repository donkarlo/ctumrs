import numpy as np
from tensorflow.keras.models import Model, Sequential,load_model
from tensorflow.keras.layers import Dense

class Autoencoder:
    @staticmethod
    def loadEncoder(pathToEncoder: str):
        encoder = load_model(pathToEncoder)
        return encoder

    def __init__(self,obss:np.ndarray,latentDim,epochs):
        self.__obss = obss
        self.__latentDim = latentDim
        self.__epochs = epochs
        self.__batchSize = 32
        self.__rangesDim = 720
        self.__autoencoder = None

    def __getFittedAutoencoder(self)->Model:
        if self.__autoencoder is None:
            self.__encoder = Sequential([
                Dense(512, activation='relu', input_shape=(self.__rangesDim ,)),
                Dense(256, activation='relu'),
                Dense(128, activation='relu'),
                Dense(64, activation='relu'),
                Dense(32, activation='relu'),
                Dense(self.__latentDim, activation='relu')
            ])

            self.__decoder = Sequential([
                Dense(32, activation='relu', input_shape=(self.__latentDim,)),
                Dense(64, activation='relu'),
                Dense(128, activation='relu'),
                Dense(256, activation='relu'),
                Dense(512, activation='relu'),
                Dense(self.__rangesDim , activation=None)
            ])
            self.__autoencoder = Model(inputs=self.__encoder.input
                                       , outputs=self.__decoder(self.__encoder.output))
            self.__autoencoder.compile(loss='mse', optimizer='adam')

            print("Fitting the auto encoder ...")
            modelHistory = self.__autoencoder.fit(self.__obss
                                           , self.__obss
                                           , epochs=self.__epochs
                                           , batch_size=self.__batchSize
                                           , verbose=0)

        return self.__autoencoder




    def saveFittedAutoencoder(self, pathToAutoencoder:str):
        self.__getFittedAutoencoder().save(filepath=pathToAutoencoder)

    def saveFittedEncoder(self, pathToEncoder:str):
        self.getFittedEncoder().save(filepath=pathToEncoder)

    def saveFittedDecoder(self, pathToDecoder:str):
        self.getFittedDecoder().save(filepath=pathToDecoder)

    def getFittedEncoder(self)->Sequential:
        self.__getFittedAutoencoder()
        return self.__encoder

    def getFittedDecoder(self)->Sequential:
        self.__getFittedAutoencoder()
        return self.__decoder

    def getPredictedLowDimObss(self,obss:np.ndarray)->np.ndarray:
        print("Building the latent space of normal scenario ladar data ...")
        return self.getFittedEncoder().predict(obss)