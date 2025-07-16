import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Dict

import numpy as np
import torch


class SensorType(Enum):
    """
    Enum class for sensor types. Contains a combination of sensor and location on body.
    When left/right is not specified, default is right.
    Consider ACC and GYRO on waist and hip as the same type because they shouldn't differ much.
    """
    ACC_BACK_LOWER = auto()
    ACC_THIGH_RIGHT = auto()
    ACC_WRIST_RIGHT = auto()
    ACC_WRIST_LEFT = auto()
    ACC_ANKLE_LEFT = auto()
    ACC_ANKLE_RIGHT = auto()
    ACC_CHEST = auto()
    ACC_TROUSER_FRONT_POCKET_RIGHT = auto()
    ACC_TROUSER_FRONT_POCKET_LEFT = auto()
    ACC_HIP_LEFT = auto()
    ACC_HIP_RIGHT = auto()
    EDA = auto()
    HR = auto()
    BVP = auto()
    TEMP_BODY = auto()
    TEMP_SKIN = auto()
    IBI = auto()
    GYRO_WRIST_RIGHT = auto()
    GYRO_WRIST_LEFT = auto()
    GYRO_ANKLE_LEFT = auto()
    GYRO_ANKLE_RIGHT = auto()
    GYRO_CHEST = auto()
    GYRO_TROUSER_FRONT_POCKET = auto()
    GYRO_HIP_RIGHT = auto()
    ECG_CHEST = auto()


class ActivityLabel(Enum):
    """
    Enum class for human activity labels.
    """
    WALKING = auto()
    WALKING_UPSTAIR = auto()
    WALKING_DOWNSTAIR = auto()
    JOGGING = auto()
    JUMPING = auto()
    SHUFFLING = auto()
    DRIVING = auto()
    RUNNING = auto()
    RESTING = auto()
    STANDING = auto()
    SITTING = auto()
    CYCLING = auto()


# Specifies how many axes each sensor type have
SENSOR_AXES_MAP = {
    SensorType.ACC_BACK_LOWER: 3,
    SensorType.ACC_THIGH_RIGHT: 3,
    SensorType.ACC_WRIST_RIGHT: 3,
    SensorType.ACC_WRIST_LEFT: 3,
    SensorType.ACC_ANKLE_LEFT: 3,
    SensorType.ACC_ANKLE_RIGHT: 3,
    SensorType.ACC_CHEST: 3,
    SensorType.ACC_TROUSER_FRONT_POCKET_RIGHT: 3,
    SensorType.ACC_TROUSER_FRONT_POCKET_LEFT: 3,
    SensorType.ACC_HIP_LEFT: 3,
    SensorType.ACC_HIP_RIGHT: 3,
    SensorType.GYRO_WRIST_RIGHT: 3,
    SensorType.GYRO_WRIST_LEFT: 3,
    SensorType.GYRO_ANKLE_LEFT: 3,
    SensorType.GYRO_ANKLE_RIGHT: 3,
    SensorType.GYRO_CHEST: 3,
    SensorType.GYRO_TROUSER_FRONT_POCKET: 3,
    SensorType.GYRO_HIP_RIGHT: 3,

    SensorType.ECG_CHEST: 2,

    SensorType.EDA: 1,
    SensorType.HR: 1,
    SensorType.BVP: 1,
    SensorType.TEMP_BODY: 1,
    SensorType.TEMP_SKIN: 1,
    SensorType.IBI: 1,
}


@dataclass
class DataPoint:
    """
    A dataclass to hold a single sample of sensor data and the label.
    """
    sensors: Dict[SensorType, np.ndarray]
    label: ActivityLabel

    def __post_init__(self):
        """Validates the shape of sensor data after initialization."""
        for sensor_type, data in self.sensors.items():
            if sensor_type not in SENSOR_AXES_MAP:
                raise ValueError(f"Sensor type {sensor_type} is not supported.")

            expected_axes = SENSOR_AXES_MAP[sensor_type]
            if data.ndim != 2 or data.shape[0] != expected_axes:
                raise ValueError(f"Incorrect shape for sensor type {sensor_type.name}."
                                 f"Expected ({expected_axes}, n_timesteps) but got {data.shape}")


class HARBaseDataset(torch.utils.data.Dataset, ABC):
    def __init__(
            self,
            dir_name: str,
            root_dir="data",
            test_mode=False,
    ):
        """
        Super class for all datasets. Handles everything from downloading the dataset to loading into DataPoint.
        """
        self.dir_name = dir_name
        self.root_dir = root_dir
        self.test_mode = test_mode

        full_path = os.path.join(root_dir, dir_name)
        is_downloaded = os.path.isdir(full_path)
        if not is_downloaded:
            self.download_data()

        self.data = self.read_all(root_dir, dir_name)
        self.size = len(self.data)

    @abstractmethod
    def download_data(self):
        pass


    @abstractmethod
    def read_all(self, root_path, dir_name) -> List[DataPoint]:
        pass

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return self.size


