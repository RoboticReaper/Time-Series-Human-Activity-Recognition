import numpy as np
from collections import defaultdict
from datasets import DataPoint, Sensor, ActivityLabel
import torch

class ZScoreNormalizer:
    """
    A callable class to apply Z-score normalization using pre-computed stats.
    """
    def __init__(self, mean, std):
        # Add a small epsilon to std to prevent division by zero
        self.mean = mean
        self.std = std + 1e-8

    def __call__(self, data_point: DataPoint):
        """Applies normalization to a DataPoint's sensors."""
        for sensor_type, data in data_point.sensors.items():
            if sensor_type in self.mean:
                data_point.sensors[sensor_type] = (data - self.mean[sensor_type]) / self.std[sensor_type]
        return data_point


def compute_stats(dataset: torch.utils.data.Dataset):
    """Computes mean and std for each sensor type from a dataset."""
    sums = defaultdict(float)
    sq_sums = defaultdict(float)
    counts = defaultdict(int)

    print("Computing normalization stats from training data...")
    for i in range(len(dataset)):
        data_point = dataset[i]
        for sensor_type, data in data_point.sensors.items():
            sums[sensor_type] += np.sum(data, axis=1)
            sq_sums[sensor_type] += np.sum(data ** 2, axis=1)
            counts[sensor_type] += data.shape[1]

    mean = {st: sums[st] / counts[st] for st in sums}
    std = {st: np.sqrt(sq_sums[st] / counts[st] - mean[st] ** 2) for st in sums}

    # Reshape for broadcasting
    for st in mean:
        mean[st] = mean[st].reshape(-1, 1)
        std[st] = std[st].reshape(-1, 1)

    print("Stats computed.")
    return {'mean': mean, 'std': std}