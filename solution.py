import os
import typing
from sklearn.gaussian_process.kernels import *
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib import cm


# Docker use as CMD admin:
# C:\Windows\System32\dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
# C:\Windows\System32\dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
# cd C:\Users\paesc\OneDrive\docs\projects\probabilistic-artificial-intelligence-projects\1_gaussian_progress_regression
# docker build --tag task1 .;docker run --rm -v "%cd%:/results" task1

# Set `EXTENDED_EVALUATION` to `True` in order to visualize your predictions.
EXTENDED_EVALUATION = False
EVALUATION_GRID_POINTS = 300  # Number of grid points used in extended evaluation

# Cost function constants
COST_W_UNDERPREDICT = 50.0
COST_W_NORMAL = 1.0

### PROMPT
# I want to model the distribution of air quality (PM2.5) with a gaussian process and have access to unevenly distributed data. My training data has the coordinatey (2D), as well as a binary variable indicating whether the measurement was taken in a residental area, as well as the response (PM2.5). We must not underpredict the PM2.5 value for residental areas
# By now I have set up the following properties: 
# - the loss function weights underpredictions of residence areas with weight 50 to rural area weight 1
# - for model training I multiplied all PM2.5 values with a factor of 50
# - i'm using skikit learn for my object, and so far I have chosen the Matern() + WhiteNoise Kernel

# Quickly analysie my setup and share our informed opinion briefly and I will follow up with further questions.

###

class Model(object):
    """
    Model for this task.
    """

    def __init__(self):
        """
        Initialize the Gaussian Process with the appropriate kernel.
        """

        self.matern_length_scale = 1.
        self.rng = np.random.default_rng(seed=0)
        self.kernel =   DotProduct(sigma_0=1.) + \
                        ConstantKernel(constant_value=1.) * \
                        Matern(length_scale=1., length_scale_bounds=(1e-05, 100000.0), nu=1.5) + \
                        WhiteKernel(noise_level = 1.)
        # self.kernel = DotProduct(sigma_0=1.) + RationalQuadratic(length_scale=1.0, alpha=0.1) + WhiteKernel(noise_level=0.5)
        # self.kernel = DotProduct(sigma_0=1.0) + ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel()
        self.gp = GaussianProcessRegressor(kernel=self.kernel)
        
        # We will store the mean and std of the targets for normalization and denormalization
        self.target_mean = None
        self.target_std = None
        # Introduce a latent variable to control how much of the uncertainty we add
        self.std_multiplier = 1

        # Lock the undersampling method to KMeans
        self.undersampling_method = self.kmeans_undersampling
        self.n_samples = 8000

    def normalize_targets(self, targets: np.ndarray):
        """
        Normalize the target variable (PM2.5) using Z-score normalization.
        :param targets: 1d NumPy array of targets to normalize
        :return: Normalized targets
        """
        self.target_mean = np.mean(targets)
        self.target_std = np.std(targets)
        return (targets - self.target_mean) / self.target_std

    def denormalize_targets(self, normalized_targets: np.ndarray):
        """
        Denormalize the target variable back to its original scale.
        :param normalized_targets: 1d NumPy array of normalized targets to denormalize
        :return: Denormalized targets
        """
        return normalized_targets * self.target_std + self.target_mean

    def denormalize_std(self, normalized_std: np.ndarray):
        """
        Denormalize the standard deviation back to its original scale.
        :param normalized_std: 1d NumPy array of normalized standard deviations to denormalize
        :return: Denormalized standard deviations
        """
        return normalized_std * self.target_std
    
    def train_model(self, train_targets: np.ndarray, train_coordinates: np.ndarray, train_area_flags: np.ndarray):
        """
        Fit the GP model on the given training data, with normalization of the targets.
        :param train_coordinates: 2d NumPy array of shape (NUM_SAMPLES, 2)
        :param train_targets: 1d NumPy array of shape (NUM_SAMPLES,)
        :param train_area_flags: Binary variable denoting whether the 2D training point is in the residential area (1) or not (0)
        """

        # Normalize the target values (the rest is already normalized)
        normalized_train_targets = self.normalize_targets(train_targets)

        # Undersample the data
        train_coordinates, train_area_flags, normalized_train_targets = self.undersampling_method(train_coordinates, train_area_flags, normalized_train_targets, self.n_samples)

        # Fit the GP model on normalized targets
        self.gp.fit(train_coordinates, normalized_train_targets)

    def generate_predictions(self, test_coordinates: np.ndarray, test_area_flags: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict the pollution concentration for a given set of city_areas.
        :param test_coordinates: 2d NumPy array of shape (NUM_SAMPLES, 2)
        :param test_area_flags: Binary variable denoting residential areas (1) or not (0)
        :return: Tuple of predictions, GP posterior mean, and GP posterior stddev
        """

        # Use GP to estimate the posterior mean and stddev for the test coordinates
        gp_mean, gp_std = self.gp.predict(test_coordinates, return_std=True)

        # Denormalize the GP mean to get the predictions in the original scale
        denormalized_gp_mean, denormalized_gp_std = self.denormalize_targets(gp_mean), self.denormalize_std(gp_std)

        # Overpredict in residence areas by adding the standard deviation to the mean
        predictions = np.where(test_area_flags, denormalized_gp_mean + self.std_multiplier * denormalized_gp_std, denormalized_gp_mean)

        # Return the predictions, posterior mean, and posterior stddev
        return predictions, denormalized_gp_mean, denormalized_gp_std
    
    @staticmethod
    def kmeans_undersampling(data: np.ndarray, area: np.array, responses: np.ndarray, n_samples: int = 2000):
        """
        Perform K-Means clustering to undersample 2D data, returning representative points with their responses.
        
        Parameters:
            data (np.ndarray): The input dataset as a 2D NumPy array of shape (num_samples, 2).
            responses (np.ndarray): The response values associated with each sample, as a 1D array.
            n_clusters (int): The number of clusters to form. Default is 2000.
        
        Returns:
            np.ndarray: A reduced dataset containing the representative samples from each cluster.
            np.ndarray: The corresponding response values for the selected representative samples.
        """
        
        # Initialize the K-Means model
        kmeans = KMeans(n_clusters=n_samples, random_state=0, n_init="auto")
        
        # Fit the model to the data
        kmeans.fit(data)
        
        # Get the centroids of the clusters
        centroids = kmeans.cluster_centers_
        
        # Vectorized distance calculation between centroids and all data points
        # Resulting distances will be a matrix of shape (n_clusters, num_samples)
        distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)

        # Find the index of the closest point for each centroid
        representative_indices = np.argmin(distances, axis=0)
        
        # Use the indices to get the representative samples and their responses
        
        return data[representative_indices], area[representative_indices], responses[representative_indices]
    
    @staticmethod
    def random_undersampling(data: np.ndarray, area: np.array, responses: np.ndarray, n_samples: int = 2000):
        """
        Perform random undersampling to select a subset of the data.
        
        Parameters:
            data (np.ndarray): The input dataset as a 2D NumPy array of shape (num_samples, 2).
            responses (np.ndarray): The response values associated with each sample, as a 1D array.
            n_samples (int): The number of samples to select. Default is 2000.
        
        Returns:
            np.ndarray: A reduced dataset containing the selected samples.
            np.ndarray: The corresponding response values for the selected samples.
        """
        
        # Randomly select n_samples indices
        indices = np.random.choice(data.shape[0], n_samples, replace=False)
        
        # Use the indices to get the selected samples and their responses

        return data[indices], area[indices], responses[indices]


def calculate_cost(ground_truth: np.ndarray, predictions: np.ndarray, area_flags: np.ndarray) -> float:
    """
    Calculates the cost of a set of predictions.

    :param ground_truth: Ground truth pollution levels as a 1d NumPy float array
    :param predictions: Predicted pollution levels as a 1d NumPy float array
    :param area_flags: city_area info for every sample in a form of a bool array (NUM_SAMPLES,)
    :return: Total cost of all predictions as a single float
    """
    assert ground_truth.ndim == 1 and predictions.ndim == 1 and ground_truth.shape == predictions.shape

    # Unweighted cost
    cost = (ground_truth - predictions) ** 2
    weights = np.ones_like(cost) * COST_W_NORMAL

    # Case i): underprediction
    mask = (predictions < ground_truth) & [bool(area_flag) for area_flag in area_flags]
    weights[mask] = COST_W_UNDERPREDICT

    # Weigh the cost and return the average
    return np.mean(cost * weights)


def check_within_circle(coordinate, circle_parameters):
    """
    Checks if a coordinate is inside a circle.
    :param coordinate: 2D coordinate
    :param circle_parameters: 3D coordinate of the circle center and its radius
    :return: True if the coordinate is inside the circle, False otherwise
    """
    return (coordinate[0] - circle_parameters[0])**2 + (coordinate[1] - circle_parameters[1])**2 < circle_parameters[2]**2

# You don't have to change this function 
def identify_city_area_flags(grid_coordinates):
    """
    Determines the city_area index for each coordinate in the visualization grid.
    :param grid_coordinates: 2D coordinates of the visualization grid
    :return: 1D array of city_area indexes
    """
    # Circles coordinates
    circles = np.array([[0.5488135, 0.71518937, 0.17167342],
                    [0.79915856, 0.46147936, 0.1567626 ],
                    [0.26455561, 0.77423369, 0.10298338],
                    [0.6976312,  0.06022547, 0.04015634],
                    [0.31542835, 0.36371077, 0.17985623],
                    [0.15896958, 0.11037514, 0.07244247],
                    [0.82099323, 0.09710128, 0.08136552],
                    [0.41426299, 0.0641475,  0.04442035],
                    [0.09394051, 0.5759465,  0.08729856],
                    [0.84640867, 0.69947928, 0.04568374],
                    [0.23789282, 0.934214,   0.04039037],
                    [0.82076712, 0.90884372, 0.07434012],
                    [0.09961493, 0.94530153, 0.04755969],
                    [0.88172021, 0.2724369,  0.04483477],
                    [0.9425836,  0.6339977,  0.04979664]])
    
    area_flags = np.zeros((grid_coordinates.shape[0],))

    for i,coordinate in enumerate(grid_coordinates):
        area_flags[i] = any([check_within_circle(coordinate, circ) for circ in circles])

    return area_flags

def execute_extended_evaluation(model: Model, output_dir: str = '/results'):
    """
    Visualizes the predictions of a fitted model.
    :param model: Fitted model to be visualized
    :param output_dir: Directory in which the visualizations will be stored
    """
    print('Performing extended evaluation')

    # Visualize on a uniform grid over the entire coordinate system
    grid_lat, grid_lon = np.meshgrid(
        np.linspace(0, EVALUATION_GRID_POINTS - 1, num=EVALUATION_GRID_POINTS) / EVALUATION_GRID_POINTS,
        np.linspace(0, EVALUATION_GRID_POINTS - 1, num=EVALUATION_GRID_POINTS) / EVALUATION_GRID_POINTS,
    )
    visualization_grid = np.stack((grid_lon.flatten(), grid_lat.flatten()), axis=1)
    grid_area_flags = identify_city_area_flags(visualization_grid)
    
    # Obtain predictions, means, and stddevs over the entire map
    predictions, gp_mean, gp_stddev = model.generate_predictions(visualization_grid, grid_area_flags)
    predictions = np.reshape(predictions, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))
    gp_mean = np.reshape(gp_mean, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))

    vmin, vmax = 0.0, 65.0

    # Plot the actual predictions
    fig, ax = plt.subplots()
    ax.set_title('Extended visualization of task 1')
    im = ax.imshow(predictions, vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax)

    # Save figure to pdf
    figure_path = os.path.join(output_dir, 'extended_evaluation.pdf')
    fig.savefig(figure_path)
    print(f'Saved extended evaluation to {figure_path}')

    plt.show()


def extract_area_information(train_x: np.ndarray, test_x: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts the city_area information from the training and test features.
    :param train_x: Training features
    :param test_x: Test features
    :return: Tuple of (training features' 2D coordinates, training features' city_area information,
        test features' 2D coordinates, test features' city_area information)
    """
    train_coordinates = np.zeros((train_x.shape[0], 2), dtype=float)
    train_area_flags = np.zeros((train_x.shape[0],), dtype=bool)
    test_coordinates = np.zeros((test_x.shape[0], 2), dtype=float)
    test_area_flags = np.zeros((test_x.shape[0],), dtype=bool)

    # Extract the city_area information from the training and test features
    train_coordinates, train_area_flags = train_x[:, :2], train_x[:, 2] == 1
    test_coordinates, test_area_flags = test_x[:, :2], test_x[:, 2] == 1

    assert train_coordinates.shape[0] == train_area_flags.shape[0] and test_coordinates.shape[0] == test_area_flags.shape[0]
    assert train_coordinates.shape[1] == 2 and test_coordinates.shape[1] == 2
    assert train_area_flags.ndim == 1 and test_area_flags.ndim == 1

    return train_coordinates, train_area_flags, test_coordinates, test_area_flags

# you don't have to change this function
def main():
    # Add the cwd to the path
    # Set os.getwd + 1_gaussian_progress_regression to path
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Set the cwd to the script directory
    os.chdir(script_dir)

    # Load the training dateset and test features
    train_x = np.loadtxt('data/train_x.csv', delimiter=',', skiprows=1)
    train_y = np.loadtxt('data/train_y.csv', delimiter=',', skiprows=1)
    test_x = np.loadtxt('data/test_x.csv', delimiter=',', skiprows=1)

    # Extract the city_area information
    train_coordinates, train_area_flags, test_coordinates, test_area_flags = extract_area_information(train_x, test_x)
    
    # Fit the model
    print('Training model')
    model = Model()
    model.train_model(train_y, train_coordinates, train_area_flags)

    # Predict on the test features
    print('Predicting on test features')
    predictions = model.generate_predictions(test_coordinates, test_area_flags)
    print(predictions)

    if EXTENDED_EVALUATION:
        execute_extended_evaluation(model, output_dir='.')


if __name__ == "__main__":
    main()
