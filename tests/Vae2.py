import pickle
import numpy as np
import matplotlib.pyplot as plt

from keras import backend as K
from keras.layers import Input, Dense, Lambda, Layer, Add, Multiply
from keras.models import Model, Sequential
from matplotlib.patches import Ellipse
from sklearn.preprocessing import MinMaxScaler
from matplotlib.colors import to_rgba

sensorObssLen = 36000
robotsNum = 2
intermediateDim = 256
latentDim = 2
batchSize = 100
epochs = 25
epsilonStd = 1.0
activeSencor="gps"
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
    for robot1TimeLidarValVelObsCounter, robot1TimeLidarValVelObs in enumerate(robot1NpTimeLidarValVelObss):
        if robot1TimeLidarValVelObsCounter >= len(robot2NpTimeLidarValVelObss):
            break
        robot2TimeLidarValVelObs = robot2NpTimeLidarValVelObss[robot1TimeLidarValVelObsCounter]
        combinedRobotsTimeLidarValVelObs = [robot1TimeLidarValVelObs, robot2TimeLidarValVelObs]
        combinedRobotsTimeLidarValVelObss.append(combinedRobotsTimeLidarValVelObs)
    npCombinedRobotsTimeLidarValVelObss = np.asarray(combinedRobotsTimeLidarValVelObss)

    return npCombinedRobotsTimeGpsVelsObss, npCombinedRobotsTimeLidarValVelObss


npCombinedRobotsTimeGpsValVelObss, npCombinedRobotsTimeLidarValVelObss = getTrainingSensoryData()

if activeSencor=="gps":
    sensorValDim = 3
    npCombinedRobotsTimeSensorValObss = npCombinedRobotsTimeGpsValVelObss[0:sensorObssLen, :, 0:sensorValDim + 1]
elif activeSencor=="lidar":
    sensorValDim = 720
    npCombinedRobotsTimeSensorValObss = npCombinedRobotsTimeLidarValVelObss[0:sensorObssLen, :, 0:sensorValDim + 1]


robotsNumValObsShape = npCombinedRobotsTimeSensorValObss.shape[1] * (npCombinedRobotsTimeSensorValObss.shape[2] - 1)
npSensorValObssLen = npCombinedRobotsTimeSensorValObss.shape[0]
npSensorValFlatObss = npCombinedRobotsTimeSensorValObss[:, :, 1:].reshape((npSensorValObssLen, robotsNumValObsShape))
sensorScaler = MinMaxScaler()
npSensorValFlatObssScaled = sensorScaler.fit_transform(npSensorValFlatObss)


def nll(y_true, y_pred):
    """ Negative log likelihood (Bernoulli). """
    return K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)


class KLDivergenceLayer(Layer):
    """ Identity transform layer that adds KL divergence
    to the final model loss.
    """

    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        super(KLDivergenceLayer, self).__init__(*args, **kwargs)

    def call(self, inputs):
        mu, log_var = inputs
        kl_batch = - .5 * K.sum(1 + log_var - K.square(mu) - K.exp(log_var), axis=-1)
        kl_coefficient = 0.01
        self.add_loss(kl_coefficient*K.mean(kl_batch), inputs=inputs)
        return inputs


decoder = Sequential([
    Dense(intermediateDim, input_dim=latentDim, activation='relu'),
    Dense(2 * sensorValDim, activation='sigmoid')
])

#2 here is for mean and variance
x = Input(shape=(2 * sensorValDim,))
h = Dense(intermediateDim, activation='relu')(x)

z_mu = Dense(latentDim)(h)
z_log_var = Dense(latentDim)(h)

z_mu, z_log_var = KLDivergenceLayer()([z_mu, z_log_var])
z_sigma = Lambda(lambda t: K.exp(.5 * t))(z_log_var)

eps = Input(tensor=K.random_normal(stddev=epsilonStd, shape=(K.shape(x)[0], latentDim)))
z_eps = Multiply()([z_sigma, eps])
z = Add()([z_mu, z_eps])

x_pred = decoder(z)

vae = Model(inputs=[x, eps], outputs=x_pred)
vae.compile(optimizer='rmsprop', loss=nll)



# Train the VAE
history = vae.fit([npSensorValFlatObssScaled, np.random.normal(size=(sensorObssLen, latentDim))],
        npSensorValFlatObssScaled,
        shuffle=True,
        epochs=epochs,
        batch_size=batchSize,
        validation_split=0.2)

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Create an encoder model
encoder = Model(x, z_mu)

# Display a 2D plot of the digit classes in the latent space
z_test = encoder.predict(npSensorValFlatObssScaled, batch_size=batchSize)
plt.figure(figsize=(12, 12))
plt.scatter(z_test[:, 0], z_test[:, 1], alpha=0.4)
plt.xlabel('Latent Dimension 1')
plt.ylabel('Latent Dimension 2')
plt.title('2D Latent Space Visualization')
plt.show()


def visualize_random_reconstructions(model, data, num_samples=10):
    for i in range(num_samples):
        # Choose a random index
        random_index = np.random.randint(0, len(data))

        # Select the actual data
        x_actual = data[random_index].reshape(1, robotsNum * sensorValDim)

        # Reconstruct the data
        x_reconstructed = model.predict([x_actual, np.random.normal(size=(1, latentDim))])

        # Plotting the actual and reconstructed data in one graph with alpha
        plt.figure(figsize=(8, 4))
        plt.plot(x_actual.flatten(), color='blue', marker='o', linestyle='-', label='Actual Data', alpha=0.4)
        plt.plot(x_reconstructed.flatten(), color='orange', marker='o', linestyle='-', label='Reconstructed Data', alpha=0.4)
        plt.title(f'Sample {i + 1} - Actual and Reconstructed')
        plt.xlabel('Component Number')
        plt.ylabel('Component Value')
        plt.legend()
        plt.show()

# Visualize 10 random reconstructions in separate plots
visualize_random_reconstructions(vae, npSensorValFlatObssScaled, num_samples=10)


# Create an encoder model
encoder = Model(x, [z_mu, z_log_var])

# Plot the latent points and variance ellipsoids
# Number of random points to visualize
num_points = 1000
random_indices = np.random.choice(len(npSensorValFlatObssScaled), size=num_points, replace=False)

# Select random points from x_train
x_random = npSensorValFlatObssScaled[random_indices]

# Get the latent means and log variances for the selected points
z_means, z_log_vars = encoder.predict(x_random)

# Plot the latent points and variance ellipsoids
plt.figure(figsize=(12, 12))

# Plot variance ellipsoids
for i in range(num_points):
    cov_matrix = np.diag(np.exp(z_log_vars[i, :]))
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
    ellipse = Ellipse(z_means[i, :], 2 * np.sqrt(5.991 * eigenvalues[0]), 2 * np.sqrt(5.991 * eigenvalues[1]),
                      angle=angle, alpha=0.1, color='yellow', zorder=1)  # Set zorder to 1
    plt.gca().add_patch(ellipse)
    # Ellipse with a pale boundary line
    boundary_ellipse = Ellipse(z_means[i, :], 2.2 * np.sqrt(5.991 * eigenvalues[0]),
                               2.2 * np.sqrt(5.991 * eigenvalues[1]),
                               angle=angle, edgecolor=to_rgba('orange', alpha=0.5), facecolor='none', linewidth=1.5, zorder=2)
    plt.gca().add_patch(boundary_ellipse)

# Plot means
plt.scatter(z_means[:, 0], z_means[:, 1], color='red', marker='o', label='Latent Mean', zorder=3)  # Set zorder to 2

plt.xlabel('Latent Dimension 1')
plt.ylabel('Latent Dimension 2')
plt.title('Latent Space Visualization with Variance Ellipsoids')
plt.legend()
plt.show()