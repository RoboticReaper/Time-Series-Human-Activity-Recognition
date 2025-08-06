from collections import defaultdict

import torch
from tqdm import tqdm
from sensors import SensorFrequency, Sensor
from typing import Dict, Type



class STFTTransform:
    """Apply STFT transform to get spectrogram"""
    def __init__(self, n_fft, hop_length, win_length):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

    def __call__(self, sensor_data: torch.Tensor):
        """
        Input is a tensor representing a window of sensor data.
        Expect shape (axes, num_measurements)
        """
        window = torch.hann_window(self.win_length, device=sensor_data.device)

        stft_complex = torch.stft(
            sensor_data,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            return_complex=True,
            center=True
        )

        magnitude_spectrogram = torch.abs(stft_complex)

        return magnitude_spectrogram


class StatFeaturesTransform:
    def __call__(self, data: torch.Tensor):
        """
        Compute statistical features on the last dimension.
        Expect input to be shape (axes, num_measurements).
        Should be used for low frequency sensors.

        Returns a tensor of shape (axes, 6). The 6 features are mean, std, min, max, median, iqr
        """
        if data.ndim != 2:
            raise ValueError("Expected data dimension to be 2: (axes, num_measurements)"
                             f"got {data.shape}")

        mean = torch.mean(data, dim=-1)
        std = torch.std(data, dim=-1)
        min_val, _ = torch.min(data, dim=-1)
        max_val, _ = torch.max(data, dim=-1)
        median_val, _ = torch.median(data, dim=-1)

        q = torch.tensor([0.25, 0.75], device=data.device)

        # Calculate Q1 and Q3 along the last dimension
        quantiles = torch.quantile(data, q, dim=-1)
        # quantiles shape: (quantiles = 2, axes)

        quantiles = quantiles.permute(1, 0)

        q1 = quantiles[..., 0]
        q3 = quantiles[..., 1]
        iqr = q3 - q1

        features_tuple = (mean, std, min_val, max_val, median_val, iqr)
        return torch.stack(features_tuple, dim=-1)


class BaseZScoreNormalizer:
    """
    Base class for applying Z-score normalization using pre-computed stats.
    It handles the core normalization logic and initial validation.
    """

    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        """
        Initializes the normalizer with mean and standard deviation.

        Args:
            mean (torch.Tensor): A 1D tensor of mean values.
            std (torch.Tensor): A 1D tensor of standard deviation values.
        """
        if mean.ndim != 1 or std.ndim != 1:
            raise ValueError(
                "Mean and STD must be 1D tensors. "
                f"Got mean shape {mean.shape} and STD shape {std.shape}"
            )

        # Mean and std are stored here; subclasses will reshape them.
        self.epsilon = 1e-8
        self.mean = mean
        self.std = std + self.epsilon

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """
        Applies the Z-score normalization formula. This method is intended
        to be called by the subclasses after they have validated and prepared the data.

        Args:
            data (torch.Tensor): The input data to normalize.

        Returns:
            torch.Tensor: The normalized data.
        """
        return (data - self.mean) / self.std

    def __repr__(self):
        """Provides a string representation of the normalizer instance."""
        return (
            f"{self.__class__.__name__}(\n"
            f"  mean_shape={self.mean.shape},\n"
            f"  std_shape={self.std.shape},\n"
            f"  epsilon={self.epsilon}\n)"
        )

class SpectrogramZScoreNormalizer(BaseZScoreNormalizer):
    """
    Applies per-frequency-bin Z-score normalization to a spectrogram.
    Inherits from BaseZScoreNormalizer and handles spectrogram-specific tensor shaping.
    """

    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        """
        Args:
            mean (torch.Tensor): 1D tensor with shape (frequency_bins,).
            std (torch.Tensor): 1D tensor with shape (frequency_bins,).
        """
        super().__init__(mean, std)
        # Reshape for broadcasting over (axes, freq_bins, time_frames)
        self.mean = self.mean.view(1, -1, 1)
        self.std = self.std.view(1, -1, 1)

    def __call__(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Validates and normalizes a spectrogram tensor.

        Args:
            spectrogram (torch.Tensor): A 3D tensor with shape (axes, freq_bins, time_frames).

        Returns:
            torch.Tensor: The normalized spectrogram.
        """
        if spectrogram.ndim != 3:
            raise ValueError(
                "Expected spectrogram shape (axes, freq_bins, time_frames), "
                f"but got {spectrogram.shape}"
            )
        # Use the base class's __call__ for the calculation
        return super().__call__(spectrogram)

class StatFeaturesZScoreNormalizer(BaseZScoreNormalizer):
    """
    Applies per-feature Z-score normalization to statistical features.
    Inherits from BaseZScoreNormalizer and handles feature-vector-specific tensor shaping.
    """

    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        """
        Args:
            mean (torch.Tensor): 1D tensor with shape (num_features,).
            std (torch.Tensor): 1D tensor with shape (num_features,).
        """
        super().__init__(mean, std)
        # Reshape for broadcasting over (axes, num_features)
        self.mean = self.mean.view(1, -1)
        self.std = self.std.view(1, -1)

    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        """
        Validates and normalizes a statistical-features tensor.

        Args:
            features (torch.Tensor): A 2D tensor with shape (axes, num_features).

        Returns:
            torch.Tensor: The normalized features.
        """
        if features.ndim != 2:
            raise ValueError(
                f"Expected 2D feature tensor, but got shape {features.shape}"
            )
        # Use the base class's __call__ for the calculation
        return super().__call__(features)


class ConditionalTransform:
    """
    A callable transform that processes a TorchDataPoint by using different transformations based on sensor frequency.

    High frequency sensors use STFT. Low frequency sensors are converted to statistical features.
    Then they are all normalized.

    Transformation is applied in-place and cached in TorchDataPoint to avoid re-computation.
    This means if for some reason this transform is removed from a dataset,
    the data returned will still be the transformed result.
    """
    def __init__(self, stft_transform: STFTTransform,
                 stat_features_transform: StatFeaturesTransform,
                 spectrogram_normalizers: Dict[Sensor, SpectrogramZScoreNormalizer],
                 stat_features_normalizers: Dict[Sensor, StatFeaturesZScoreNormalizer]):

        self.stft_transform = stft_transform
        self.stat_features_transform = stat_features_transform

        # Dictionary of normalizer. Will pick the correct normalizer according to each sensor
        self.spectrogram_normalizers = spectrogram_normalizers
        self.stat_features_normalizers = stat_features_normalizers


    def __call__(self, datapoint):
        """datapoint is a TorchDataPoint."""
        for sensor, data in list(datapoint.sensors.items()):
            if datapoint.already_transformed.get(sensor, False):
                continue

            if sensor.frequency == SensorFrequency.HIGH:
                transformed_data = self.stft_transform(data)
                # Look up the correct normalizer for this sensor
                if sensor in self.spectrogram_normalizers:
                    normalizer = self.spectrogram_normalizers[sensor]
                    transformed_data = normalizer(transformed_data)

            elif sensor.frequency == SensorFrequency.LOW:
                transformed_data = self.stat_features_transform(data)
                # Look up the correct normalizer for this sensor
                if sensor in self.stat_features_normalizers:
                    normalizer = self.stat_features_normalizers[sensor]
                    transformed_data = normalizer(transformed_data)
            else:
                raise TypeError(f'Unknown sensor frequency category for sensor: {sensor.name}')

            datapoint.sensors[sensor] = transformed_data
            datapoint.already_transformed[sensor] = True

        return datapoint


def _fit_spectrogram_z_normalizer(dataset, transform: STFTTransform):
    """
    Calculates the mean and std for each frequency bin across the entire dataset for high frequency sensors.
    """
    all_spectrograms_by_sensor = defaultdict(list)
    print("\n\nGenerating spectrograms for stats calculation...")

    for i in tqdm(range(len(dataset))):
        data_point = dataset[i]
        for sensor, sensor_data in data_point.sensors.items():
            if sensor.frequency == SensorFrequency.HIGH:
                spectrogram = transform(sensor_data)
                all_spectrograms_by_sensor[sensor].append(spectrogram)

    stats_dict = {}
    print("Calculating spectrogram statistics per sensor...")
    for sensor, spec_list in all_spectrograms_by_sensor.items():
        full_tensor = torch.stack(spec_list, dim=0)
        mean = torch.mean(full_tensor, dim=(0, 1, 3))
        std = torch.std(full_tensor, dim=(0, 1, 3))
        stats_dict[sensor] = {'mean': mean, 'std': std}
        print(f"  - {sensor.name}: stats calculated.")

    return stats_dict


def _fit_stat_features_z_normalizer(dataset, transform: StatFeaturesTransform):
    """
    Calculates the mean and std for each statistical feature across the entire dataset for low frequency sensors.
    """
    all_features_by_sensor = defaultdict(list)
    print("\n\nGenerating statistical features for stats calculation...")
    for i in tqdm(range(len(dataset))):
        data_point = dataset[i]
        for sensor, sensor_data in data_point.sensors.items():
            if sensor.frequency == SensorFrequency.LOW:
                features = transform(sensor_data)
                all_features_by_sensor[sensor].append(features)

    stats_dict = {}
    print("Calculating feature statistics per sensor...")
    for sensor, features_list in all_features_by_sensor.items():
        # Shape of each item in list: (axes, num_features)
        full_tensor = torch.stack(features_list, dim=0)
        # Resulting shape: (num_samples, axes, num_features)

        # Calculate stats over batch and axes dimensions (0 and 1)
        mean = torch.mean(full_tensor, dim=(0, 1))
        std = torch.std(full_tensor, dim=(0, 1))
        stats_dict[sensor] = {'mean': mean, 'std': std}
        print(f"  - {sensor.name}: stats calculated.")

    return stats_dict


def _create_normalizers_dict(stats_dict: Dict[Sensor, Dict[str, torch.Tensor]], NormalizerClass: Type[BaseZScoreNormalizer]):
    """
    Creates a dictionary of initialized normalizer objects from a dictionary of stats.

    Args:
        stats_dict: A dict mapping a Sensor to its {'mean': tensor, 'std': tensor}.
        NormalizerClass: The normalizer class to instantiate (e.g., SpectrogramZScoreNormalizer).

    Returns:
        A dictionary mapping each Sensor to its initialized normalizer instance.
    """
    normalizers = {}
    for sensor, stats in stats_dict.items():
        mean = stats['mean']
        std = stats['std']
        normalizers[sensor] = NormalizerClass(mean=mean, std=std)
    return normalizers


def create_conditional_transform(dataset: torch.utils.data.Dataset, n_fft, hop_length, win_length) -> ConditionalTransform:
    """
    Creates a ConditionalTransform specific to the passed dataset to use.

    Args:
        dataset (torch.utils.data.Dataset): The training dataset that inherits from HARBaseDataset or ConcatDataset containing exclusively HARBaseDataset.
        n_fft (int): The number of frequency bins for stft transform.
        hop_length (int): The hop length for stft transform.
        win_length (int): The window length for stft transform.

    Returns:
        A ConditionalTransform callable class for the passed dataset to use.
    """

    stft_transform = STFTTransform(n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    stat_features_transform = StatFeaturesTransform()
    stft_fit = _fit_spectrogram_z_normalizer(dataset, stft_transform)
    stat_features_fit = _fit_stat_features_z_normalizer(dataset, stat_features_transform)
    spectrogram_normalizers = _create_normalizers_dict(stft_fit, SpectrogramZScoreNormalizer)
    stat_features_normalizers = _create_normalizers_dict(stat_features_fit, StatFeaturesZScoreNormalizer)

    conditional_transform = ConditionalTransform(stft_transform=stft_transform, stat_features_transform=stat_features_transform,
                                spectrogram_normalizers=spectrogram_normalizers, stat_features_normalizers=stat_features_normalizers)

    return conditional_transform
