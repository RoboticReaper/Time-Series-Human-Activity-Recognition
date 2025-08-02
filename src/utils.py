import torch
from datasets import HARBaseDataset

class SpectrogramZScoreNormalizer:
    """
    Apply per-frequency-bin Z-score normalization using pre-computed stats on a spectrogram.
    """
    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        """Mean and STD should have shape of (frequency_bins)"""
        if mean.ndim != 1 or std.ndim != 1:
            raise ValueError("Mean and STD must have 1 dimension"
                             f"got mean shape {mean.shape}, STD shape {std.shape}")

        # Reshape using view to allow broadcasting in __call__
        self.mean = mean
        self.mean_reshaped = self.mean.view(1, mean.shape[0], 1)
        # Add a small epsilon to std to prevent division by zero
        self.epsilon = 1e-8
        self.std = std + self.epsilon
        self.std_reshaped = self.std.view(1, std.shape[0], 1)

    def __call__(self, spectrogram: torch.Tensor):
        """Applies normalization to a spectrogram sensors."""
        if spectrogram.ndim != 3:
            raise ValueError("Expected spectrogram shape to be (axes, freq_bins, time_frames)"
                             f"got {spectrogram.shape}")

        return (spectrogram - self.mean_reshaped) / self.std_reshaped


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


def compute_stat_features(data: torch.Tensor):
    """
    Compute statistical features on the last dimension.
    Expect input to be shape (batch_size, axes, num_measurements).
    Should be used for low frequency sensors.

    Returns a tuple of mean, std, min, max, median, iqr
    """
    if data.ndim != 3:
        raise ValueError("Expected data dimension to be 3: (batch_size, axes, num_measurements)"
                         f"got {data.shape}")

    mean = torch.mean(data, dim=-1)
    std = torch.std(data, dim=-1)
    min_val, min_index = torch.min(data, dim=-1)
    max_val, max_index = torch.max(data, dim=-1)
    median_val, median_index = torch.median(data, dim=-1)

    q = torch.tensor([0.25, 0.75], device=data.device)

    # Calculate Q1 and Q3 along the last dimension
    quantiles = torch.quantile(data, q, dim=-1)
    # quantiles shape: (quantiles = 2, batch_size, axes)
    # Permute the shape back to (batch_size, axes, quantiles)
    quantiles = quantiles.permute(1, 2, 0)

    q1 = quantiles[..., 0]
    q3 = quantiles[..., 1]
    iqr = q3 - q1

    return mean, std, min_val, max_val, median_val, iqr


def compute_spectrogram_stats(dataset: torch.utils.data.Dataset, transform: STFTTransform):
    """
    Calculates the mean and std for each frequency bin across the entire dataset for each sensor.

    Args:
        dataset (torch.utils.data.Dataset): The training dataset that inherits from HARBaseDataset or ConcatDataset
        transform (STFTTransform): STFTTransform for the training set.

    Returns:
        A dictionary.
        Key: sensor
        Value: a tuple of (mean, std), where each is a 1D tensor of shape (num_frequency_bins).
    """
    pass



