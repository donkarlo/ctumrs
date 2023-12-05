import pickle
import numpy as np
import matplotlib.pyplot as plt

from keras import backend as K
from keras.layers import Input, Dense, Lambda, Layer, Add, Multiply
from keras.models import Model, Sequential
from matplotlib.patches import Ellipse
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import multivariate_normal

samplesNum = 6000
inputDim = 720
intermediate_dim = 256
latent_dim = 2
batch_size = 100
epochs = 5
epsilon_std = 1.0


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
        self.add_loss(K.mean(kl_batch), inputs=inputs)
        return inputs


decoder = Sequential([
    Dense(intermediate_dim, input_dim=latent_dim, activation='relu'),
    Dense(inputDim, activation='sigmoid')
])

x = Input(shape=(inputDim,))
h = Dense(intermediate_dim, activation='relu')(x)

z_mu = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)

z_mu, z_log_var = KLDivergenceLayer()([z_mu, z_log_var])
z_sigma = Lambda(lambda t: K.exp(.5 * t))(z_log_var)

eps = Input(tensor=K.random_normal(stddev=epsilon_std, shape=(K.shape(x)[0], latent_dim)))
z_eps = Multiply()([z_sigma, eps])
z = Add()([z_mu, z_eps])

x_pred = decoder(z)

vae = Model(inputs=[x, eps], outputs=x_pred)
vae.compile(optimizer='rmsprop', loss=nll)

# Load Lidar data
testSharedPath = "/home/donkarlo/Desktop/lstm/"
with open('{}/normalScenarioRobot1TimeLidarValVelObss.pkl'.format(testSharedPath), 'rb') as file:
    robot1NpTimeLidarValVelObss = pickle.load(file)

lidarValObss = robot1NpTimeLidarValVelObss[0:samplesNum, 1:inputDim + 1]
minMaxScaler = MinMaxScaler()
x_train = minMaxScaler.fit_transform(lidarValObss)  # Normalize the data (assuming it's in the range [0, 255])

# Train the VAE
vae.fit([x_train, np.random.normal(size=(samplesNum, latent_dim))],
        x_train,
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size)

# Create an encoder model
encoder = Model(x, z_mu)

# Display a 2D plot of the digit classes in the latent space
z_test = encoder.predict(x_train, batch_size=batch_size)
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
        x_actual = data[random_index].reshape(1, inputDim)

        # Reconstruct the data
        x_reconstructed = model.predict([x_actual, np.random.normal(size=(1, latent_dim))])

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
visualize_random_reconstructions(vae, x_train, num_samples=10)


# Create an encoder model
encoder = Model(x, [z_mu, z_log_var])

# Display a 2D plot of the digit classes in the latent space
z_test_mean, z_test_log_var = encoder.predict(x_train, batch_size=batch_size)
# Plot the latent space with mean and variance
plt.figure(figsize=(12, 12))
plt.scatter(z_test_mean[:, 0], z_test_mean[:, 1], alpha=0.4, label='Latent Mean')
plt.scatter(z_test_log_var[:, 0], z_test_log_var[:, 1], alpha=0.4, label='Latent Log Variance')
plt.xlabel('Latent Dimension 1')
plt.ylabel('Latent Dimension 2')
plt.title('2D Latent Space Visualization with Mean and Variance')
plt.legend()
plt.show()


def visualize_random_reconstructions2(model, data, num_samples=10):
    for i in range(num_samples):
        # Choose a random index
        random_index = np.random.randint(0, len(data))

        # Select the actual data
        x_actual = data[random_index].reshape(1, inputDim)

        # Reconstruct the data
        z_mean, z_log_var = encoder.predict(x_actual)
        eps = np.random.normal(size=(1, latent_dim))
        z = np.exp(0.5 * z_log_var) * eps + z_mean
        x_reconstructed = model.predict([x_actual, eps])

        # Plotting the actual, reconstructed data, mean, and variance in one graph with alpha
        plt.figure(figsize=(8, 4))
        plt.plot(x_actual.flatten(), color='blue', marker='o', linestyle='-', label='Actual Data', alpha=0.4)
        plt.plot(x_reconstructed.flatten(), color='orange', marker='o', linestyle='-', label='Reconstructed Data', alpha=0.4)
        plt.scatter(z_mean[:, 0], z_mean[:, 1], color='green', marker='x', label='Latent Mean')
        plt.scatter(z_log_var[:, 0], z_log_var[:, 1], color='red', marker='x', label='Latent Log Variance')
        plt.title(f'Sample {i + 1} - Actual, Reconstructed, Mean, and Log Variance')
        plt.xlabel('Component Number')
        plt.ylabel('Component Value')
        plt.legend()
        plt.show()

# Visualize 10 random reconstructions in separate plots
visualize_random_reconstructions2(vae, x_train, num_samples=10)



# Number of random points to visualize
num_points = 1000
random_indices = np.random.choice(len(x_train), size=num_points, replace=False)

# Select random points from x_train
x_random = x_train[random_indices]

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
                      angle=angle, alpha=0.05, color='red')
    plt.gca().add_patch(ellipse)

# Plot means
plt.scatter(z_means[:, 0], z_means[:, 1], color='blue', marker='o', label='Latent Mean')

plt.xlabel('Latent Dimension 1')
plt.ylabel('Latent Dimension 2')
plt.title('Latent Space Visualization with Variance Ellipsoids')
plt.legend()
plt.show()


# Modify the encoder model definition so that you can get a sample for each var and mean
z_eps = Lambda(lambda t: K.random_normal(stddev=epsilon_std, shape=(K.shape(t)[0], latent_dim)))(x)
encoder = Model(x, [z_mu, z_log_var, z_eps])

# ...

# Update how you use the encoder to get mean, log variance, and sampled point
z_mean, z_log_var, z_eps = encoder.predict(x_train, batch_size=batch_size)
z_sampled = np.exp(0.5 * z_log_var) * z_eps + z_mean

# ...

# Update the visualize_random_reconstructions function to use z_sampled
def visualize_random_reconstructions3(model, data, num_samples=10):
    for i in range(num_samples):
        # Choose a random index
        random_index = np.random.randint(0, len(data))

        # Select the actual data
        x_actual = data[random_index].reshape(1, inputDim)

        # Reconstruct the data
        z_mean, z_log_var, z_eps = encoder.predict(x_actual)
        z_sampled = np.exp(0.5 * z_log_var) * z_eps + z_mean
        x_reconstructed = model.predict([x_actual, np.random.normal(size=(1, latent_dim))])

        # Plotting the actual, reconstructed data, mean, sampled point, and log variance in one graph with alpha
        plt.figure(figsize=(8, 4))
        plt.plot(x_actual.flatten(), color='blue', marker='o', linestyle='-', label='Actual Data', alpha=0.4)
        plt.plot(x_reconstructed.flatten(), color='orange', marker='o', linestyle='-', label='Reconstructed Data', alpha=0.4)
        plt.scatter(z_mean[:, 0], z_mean[:, 1], color='green', marker='x', label='Latent Mean')
        plt.scatter(z_sampled[:, 0], z_sampled[:, 1], color='purple', marker='*', label='Sampled Point')
        plt.scatter(z_log_var[:, 0], z_log_var[:, 1], color='red', marker='x', label='Latent Log Variance')
        plt.title(f'Sample {i + 1} - Actual, Reconstructed, Mean, Sampled Point, and Log Variance')
        plt.xlabel('Component Number')
        plt.ylabel('Component Value')
        plt.legend()
        plt.show()

# Visualize 10 random reconstructions in separate plots
visualize_random_reconstructions3(vae, x_train, num_samples=10)