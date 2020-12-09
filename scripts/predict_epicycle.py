from sklearn.model_selection import train_test_split
from sklearn import metrics
import os
from pathlib import Path
import json
import numpy as np
import scipy.optimize


class DataArrangeTool:
    """DataArrangeTool class.

    It extracts appropriate parameters for the model from generated data.

    :Attributes:
        astro: str
            Indicates name of astronomical object.
    """

    def __init__(self, astro: str) -> None:
        """Construct data directory.

        :param astro: (str) Name of the object.
        :param time: (float) Time in ordinal form.
        :param dist_x: (np.float) X-axis distance between astro and earth in au.
        :param dist_y: (np.float) Y-axis distance between astro and earth in au.
        :param moon_phase: (np.float) Illuminated portion of moon.
        :param arc_min: (np.float) Ecliptic longitude in arcminute.
        """
        self.astro = astro
        self.time = []
        self.dist_x = []
        self.dist_y = []
        self.moon_phase = []
        self.arc_min = []

    @staticmethod
    def load_data() -> dict:
        """Return generated data.
        """
        data_path = os.path.join(Path(os.getcwd()).parent, 'data')
        try:
            with open(os.path.join(data_path, 'processed_data.json'), 'r') as file:
                data = json.load(file)
        except IOError:
            raise Exception("I/O error")
        return data

    def set_parameters(self) -> None:
        """Fill data directories.
        """

        data = self.load_data()
        astro_data = data['astro'][self.astro]

        # Convert distance into parametric form.
        self.dist_x = astro_data['distance'] * np.cos(astro_data['ecliptic_lon'])
        self.dist_y = astro_data['distance'] * np.sin(astro_data['ecliptic_lon'])
        self.time = data['time']['ord']

        if self.astro == 'moon':
            self.moon_phase = astro_data['phase_portion']
            self.arc_min = astro_data['arc_min']
        elif self.astro == 'sun':
            self.arc_min = astro_data['arc_min']


class PulsatingEpicycleModel(DataArrangeTool):
    """PulsatingEpicycleModel class.

    It predicts planetary motions by applying Fast Fourier Transformation (FFT).
    It is also an inheritance of DataArrangeTool class.

    :Attributes:
        astro: str
            Indicates name of astronomical object.
    """

    def __init__(self, astro: str) -> None:
        """Construct data directory for FFT analysis.

        :param covariance: (list) Covariances of optimized fourier parameters.
        :param amplitude: (float) Optimized amplitude.
        :param omega: (float) Optimized angular velocity.
        :param phi: (float) Optimized phi value.
        :param offset: (float) Optimized offset value.
        :param frequency: (float) Optimized frequency of FFT.
        :param period: (float) Optimized period of FFT.
        """
        super().__init__(astro)
        self.covariance = []
        self.amplitude = 0
        self.omega = 0
        self.phi = 0
        self.offset = 0
        self.frequency = 0
        self.period = 0
        self.x_ = []
        self.x_t = []
        self.y_ = []
        self.y_t = []

    def split_test_train(self, test_size: float = 0.2, axis: str = 'x') -> None:
        """Split data into training and testing based on test_size.

        :param test_size: Remaining ratio of test data after split
        :param axis: Which distance axis? X or Y?
        """
        if axis == 'x':
            dist = self.dist_x
        elif axis == 'y':
            dist = self.dist_y
        else:
            raise Exception("Write x or y only")

        time_, dist_ = np.array(self.time), np.array(dist)

        self.x_, self.x_t, self.y_, self.y_t = train_test_split(time_,
                                                                dist_,
                                                                test_size=test_size,
                                                                shuffle=False)

    def generate_sampling(self,
                          n: int = 200,
                          step: float = 0.1,
                          sigma: float = 1.0,
                          add_noise: bool = True) -> (np.array, np.array, np.array):
        """Transform x and y data into discrete form with n number of points.

        :param n: Number of data points.
        :param step: Steps on fourier series
        :param sigma: Noise parameter.
        :param add_noise: Determine adding noise or not.
        """

        # Set fourier index depending on n
        sample_points = np.linspace(0, len(self.x_) - 1, n // 2, dtype='int', endpoint=False)

        # Apply the index on both training data
        self.x_, self.y_ = self.x_[sample_points], self.y_[sample_points]

        if add_noise:
            self.x_t = self.x_t + sigma * np.random.random(len(self.x_t))

    @staticmethod
    def sin_function(t: int, a: float, w: float, phi: float, c: float) -> float:
        """Generate sine function.

        :param t: Time
        :param a: Amplitude
        :param w: Angular velocity
        :param phi: Phi
        :param c: Offset value
        """
        return a * np.sin(w * t + phi) + c

    def optimize_fourier(self) -> None:
        """Do fourier regression on time and distance.
        """
        # Calculate fft frequencies of data and set initial guess values.
        fft_x = np.fft.fftfreq(len(self.x_), d=(self.x_[1] - self.x_[0]))
        fft_y = abs(np.fft.fft(self.y_))
        freq_ = abs(fft_x[np.argmax(fft_y[1:]) + 1])
        amp_ = np.std(self.y_) * 2. ** 0.5
        offset_ = np.mean(self.y_)
        initial = np.array([amp_, 2 * np.pi * freq_, 0, offset_])

        # Fit data into fourier regression.
        params, self.covariance = scipy.optimize.curve_fit(
            self.sin_function,
            self.x_,
            self.y_,
            p0=initial,
            method='lm')

        self.amplitude, self.omega, self.phi, self.offset = params
        self.frequency = self.omega / (2 * np.pi)
        self.period = 1 / self.frequency

    def predict_position(self) -> np.array:
        """Calculate prediction of planetary motion.
        """
        assert (self.amplitude or self.omega or self.phi or self.offset), "Optimize First"

        return [self.sin_function(t, self.amplitude, self.omega, self.phi, self.offset) for t in self.x_t]

    @staticmethod
    def evaluate_regression(y_true: np.array, y_pred: np.array, planet: str) -> dict:
        """Calculate various metrics for the model.

        :param planet: Name of planet
        :param y_true: True output
        :param y_pred: Predicted output
        """
        evaluation = {
            planet: {
                         "Mean Absolute Error": metrics.mean_absolute_error(y_true, y_pred),
                         "Mean Squared Error": metrics.mean_squared_error(y_true, y_pred),
                         "Explained Variance Score": metrics.explained_variance_score(y_true, y_pred),
                         "Median Absolute Error": metrics.median_absolute_error(y_true, y_pred),
                         "R2 score": metrics.r2_score(y_true, y_pred),
                         "Max Error": metrics.max_error(y_true, y_pred)}}

        return evaluation
