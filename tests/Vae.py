import pickle

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler

#GPU ability
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    print("GPU is available")
    # Enable memory growth to prevent GPU memory allocation issues
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
else:
    print("GPU is not available")

samplesNum = 36000
inputDim = 720
latentDim = 2
testSharedPath = "/home/donkarlo/Desktop/lstm/"

with open('{}/normalScenarioRobot1TimeLidarValVelObss.pkl'.format(testSharedPath), 'rb') as file:
    robot1NpTimeLidarValVelObss = pickle.load(file)

lidarValObss = robot1NpTimeLidarValVelObss[0:samplesNum, 1:inputDim + 1]
scaler = MinMaxScaler()
lidarValObssNormalized = scaler.fit_transform(lidarValObss)
lidarValObssNormalizedTensor = tf.convert_to_tensor(lidarValObssNormalized, dtype=tf.float32)

# Define the VAE architecture
class Vae(tf.keras.Model):
    def __init__(self, latentDim):
        super(Vae, self).__init__()
        self.latent_dim = latentDim
        self.encoder = tf.keras.Sequential([
            layers.InputLayer(input_shape=(inputDim,)),
            layers.Dense(512, activation='relu'),
            layers.Dense(256, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(latentDim + latentDim),  # Adjusted to output both mean and logvar
        ])
        self.decoder = tf.keras.Sequential([
            layers.InputLayer(input_shape=(latentDim,)),
            layers.Dense(32, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(256, activation='relu'),
            layers.Dense(512, activation='relu'),
            layers.Dense(inputDim, activation='sigmoid')
        ])

    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.getDecodedLogIts(eps, apply_sigmoid=True)

    def getEncodedMeanAndLogVar(self, x):
        mean_logvar = self.encoder(x)
        mean, logvar = tf.split(mean_logvar, num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=tf.shape(mean))
        return eps * tf.exp(logvar * 0.5) + mean

    def getDecodedLogIts(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits

    def call(self, inputs, training=False, mask=None):
        x = inputs
        mean, logvar = self.getEncodedMeanAndLogVar(x)
        z = self.reparameterize(mean, logvar)
        x_logit = self.getDecodedLogIts(z)
        return x_logit

# Custom loss function for VAE
@tf.function
def compute_loss(model, x):
    mean, logvar = model.getEncodedMeanAndLogVar(x)
    z = model.reparameterize(mean, logvar)
    x_logit = model.getDecodedLogIts(z)

    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
    logpx_z = -tf.reduce_sum(cross_ent, axis=1)
    logpz = -0.5 * tf.reduce_sum(tf.square(z), axis=1)
    logqz_x = -0.5 * tf.reduce_sum(logvar + tf.square(mean) - tf.exp(logvar), axis=1)
    return -tf.reduce_mean(logpx_z + logpz - logqz_x)

# Training step function
#manually computes the gradients and applies them to the model's parameters.
@tf.function
def train_step(model, x, optimizer):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

# Training loop
def train_vae(model, dataset, optimizer, epochs):
    for epoch in range(epochs):
        for train_x in dataset:
            loss = train_step(model, train_x, optimizer)
        print('Epoch: {}, Loss: {:.4f}'.format(epoch + 1, loss))

        if (epoch + 1) % 10 == 0:
            # Visualize 10 samples
            visualize_samples(vae_model, lidarValObssNormalized)

def visualize_samples(model, data):
    # Choose 10 random indices for plotting
    random_indices = tf.random.uniform(shape=(10,), maxval=data.shape[0], dtype=tf.int32)

    for idx in random_indices:
        x_actual = data[idx].reshape(1, inputDim)  # Corrected line
        x_reconstructed = model.predict(x_actual)
        print("distance between rec and actual: ",np.linalg.norm(x_actual-x_reconstructed))

        # Plotting the actual data in blue
        plt.plot(x_actual.flatten(), color='blue', marker='o', linestyle='-', label='Actual Data')

        # Plotting the reconstructed data in orange
        plt.plot(x_reconstructed.flatten(), color='orange', marker='o', linestyle='-', label='Reconstructed Data')

        plt.xlabel('Component Number')
        plt.ylabel('Component Value')
        plt.title('Comparison between Actual and Reconstructed Data')
        plt.legend()
        plt.show()

# Create a dataset (you might need to adjust the batch size and other parameters)
batch_size = 32
dataset = tf.data.Dataset.from_tensor_slices(lidarValObssNormalizedTensor).shuffle(samplesNum).batch(batch_size)

# Initialize VAE and optimizer
vae_model = Vae(latentDim)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

# Train the VAE
train_vae(vae_model, dataset, optimizer, epochs=100)
