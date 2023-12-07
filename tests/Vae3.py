import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import tensorflow as tf
import keras
from keras import layers

import pickle
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np

sensorObssLen = 36000
robotsNum = 2
activeSencor = "gps"
# activeSencor="lidar"

# Load Lidar data
testSharedPath = "/home/donkarlo/Desktop/lstm/"


def getTrainingSensoryData():
    with open('{}/normalScenarioRobot1TimeGpsValVelObss.pkl'.format(testSharedPath), 'rb') as file:
        robot1NpTimeGpsValVelObss = pickle.load(file)
    with open('{}/normalScenarioRobot2TimeGpsValVelObss.pkl'.format(testSharedPath), 'rb') as file:
        robot2NpTimeGpsValVelObss = pickle.load(file)
    with open('{}/normalScenarioRobot1TimeLidarValVelObss.pkl'.format(testSharedPath), 'rb') as file:
        robot1NpTimeLidarValVelObss = pickle.load(file)
    with open('{}/normalScenarioRobot2TimeLidarValVelObss.pkl'.format(testSharedPath), 'rb') as file:
        robot2NpTimeLidarValVelObss = pickle.load(file)

    combinedRobotsTimeGpsVelsObss = []
    for robot1TimeGpsVelsObsCounter, robot1TimeGpsVelsObs in enumerate(robot1NpTimeGpsValVelObss):
        if robot1TimeGpsVelsObsCounter >= len(robot2NpTimeGpsValVelObss):
            break
        robot2TimeGpsValVelObs = robot2NpTimeGpsValVelObss[robot1TimeGpsVelsObsCounter]
        combinedRobotsTimeGpsValVelObs = [robot1TimeGpsVelsObs, robot2TimeGpsValVelObs]
        combinedRobotsTimeGpsVelsObss.append(combinedRobotsTimeGpsValVelObs)
    npCombinedRobotsTimeGpsVelsObss = np.asarray(combinedRobotsTimeGpsVelsObss)

    combinedRobotsTimeLidarValVelObss = []
    for robot1TimeLidarValVelObsCounter, robot1TimeLidarValVelObs in enumerate(
            robot1NpTimeLidarValVelObss):
        if robot1TimeLidarValVelObsCounter >= len(robot2NpTimeLidarValVelObss):
            break
        robot2TimeLidarValVelObs = robot2NpTimeLidarValVelObss[robot1TimeLidarValVelObsCounter]
        combinedRobotsTimeLidarValVelObs = [robot1TimeLidarValVelObs, robot2TimeLidarValVelObs]
        combinedRobotsTimeLidarValVelObss.append(combinedRobotsTimeLidarValVelObs)
    npCombinedRobotsTimeLidarValVelObss = np.asarray(combinedRobotsTimeLidarValVelObss)

    return npCombinedRobotsTimeGpsVelsObss, npCombinedRobotsTimeLidarValVelObss


npCombinedRobotsTimeGpsValVelObss, npCombinedRobotsTimeLidarValVelObss = getTrainingSensoryData()

if activeSencor == "gps":
    sensorValDim = 6  # Change to 6 for GPS data
    npCombinedRobotsTimeSensorValObss = npCombinedRobotsTimeGpsValVelObss[
                                       0:sensorObssLen, :, 0:sensorValDim + 1]
elif activeSencor == "lidar":
    sensorValDim = 720
    npCombinedRobotsTimeSensorValObss = npCombinedRobotsTimeLidarValVelObss[
                                       0:sensorObssLen, :, 0:sensorValDim + 1]

robotsNumValObsShape = npCombinedRobotsTimeSensorValObss.shape[1] * (
        npCombinedRobotsTimeSensorValObss.shape[2] - 1)
npSensorValObssLen = npCombinedRobotsTimeSensorValObss.shape[0]
npSensorValFlatObss = npCombinedRobotsTimeSensorValObss[:, :, 1:].reshape(
    (npSensorValObssLen, robotsNumValObsShape))
sensorScaler = MinMaxScaler()
npSensorValFlatObssScaled = sensorScaler.fit_transform(npSensorValFlatObss)

data = npSensorValFlatObssScaled

# Your distribution
latent_dim = 2
data_train, data_test = data[:-5000], data[-5000:]


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


# =============================================================================
# Encoder
# There are many valid configurations of hyperparameters,
# Here it is also doable without Dropout, regularization and BatchNorm
# =============================================================================
encoder_inputs = keras.Input(shape=(2 * sensorValDim))
x = layers.Dense(32, activation="relu")(encoder_inputs)
x = layers.Dense(64, activation="relu")(x)
x = layers.Dense(16, activation="relu")(x)
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
encoder.summary()

# =============================================================================
# Decoder
# Contrary to other Architectures we don't aim for a categorical output
# in a range of 0...Y so linear activation in the end
# NOTE: Normalizing the training data allows the use of other functions
# but I did not test that.
# =============================================================================
latent_inputs = keras.Input(shape=(latent_dim,))
x = layers.Dense(128, activation="relu")(latent_inputs)
x = layers.Dense(64, activation="relu")(x)
x = layers.Dense(32, activation="relu")(x)
decoder_outputs = layers.Dense(2 * sensorValDim, activation="sigmoid")(x)
decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
decoder.summary()

# =============================================================================
# Create a model class
# =============================================================================
class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def call(self, inputs, training=False, mask=None):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstruction = self.decoder(z)

        if training:
            return reconstruction

        return reconstruction

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)

            if len(data.shape) == 2:
                # Reshape data for binary_crossentropy calculation
                data = tf.reshape(data, [-1, 1, data.shape[1]])
                reconstruction = tf.reshape(reconstruction, [-1, 1, reconstruction.shape[1]])

            reconstruction_loss = tf.reduce_mean(
                keras.losses.binary_crossentropy(tf.reshape(data, tf.shape(reconstruction)), reconstruction)
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }


# =============================================================================
# Training
# EarlyStopping is strongly recommended here
# but sometimes gets stuck early
# Increase the batch size if there are more samples available!
# =============================================================================
vae = VAE(encoder, decoder)
callback = tf.keras.callbacks.EarlyStopping(monitor='loss',
                                            patience=10,
                                            restore_best_weights=False)

vae.compile(optimizer=tf.keras.optimizers.Adam())
vae.fit(data_train, epochs=20, batch_size=11, callbacks=[callback])

# Visualize 10 random reconstructions in separate plots
def visualize_reconstructions(model, data, num_samples=10):
    for i in range(num_samples):
        # Choose a random index
        random_index = np.random.randint(0, len(data))

        # Select the actual data
        x_actual = data[random_index].reshape(1, -1)

        # Reconstruct the data
        x_reconstructed = model.predict(x_actual)

        # Plotting the actual and reconstructed data in one graph
        plt.figure(figsize=(8, 4))
        plt.plot(x_actual.flatten(), color='blue', marker='o', linestyle='-', label='Actual Data')
        plt.plot(x_reconstructed.flatten(), color='orange', marker='o', linestyle='-', label='Reconstructed Data')
        plt.title(f'Sample {i + 1} - Actual and Reconstructed')
        plt.xlabel('Component Number')
        plt.ylabel('Component Value')
        plt.legend()
        plt.show()

visualize_reconstructions(vae, data_test, num_samples=10)
