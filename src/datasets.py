import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Dict

import numpy as np
import torch
import wget
import tempfile
import zipfile
import pandas as pd




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
    LYING = auto()


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


# ASSUMPTIONS HAVE BEEN MADE
# Specifies the standard sampling rate for each sensor
TARGET_SAMPLING_RATE = {
    SensorType.ACC_BACK_LOWER: 30,
    SensorType.ACC_THIGH_RIGHT: 30,
    SensorType.ACC_WRIST_RIGHT: 30,
    SensorType.ACC_WRIST_LEFT: 30,
    SensorType.ACC_ANKLE_LEFT: 30,
    SensorType.ACC_ANKLE_RIGHT: 30,
    SensorType.ACC_CHEST: 30,
    SensorType.ACC_TROUSER_FRONT_POCKET_RIGHT: 30,
    SensorType.ACC_TROUSER_FRONT_POCKET_LEFT: 30,
    SensorType.ACC_HIP_LEFT: 30,
    SensorType.ACC_HIP_RIGHT: 30,
    SensorType.GYRO_WRIST_RIGHT: 30,
    SensorType.GYRO_WRIST_LEFT: 30,
    SensorType.GYRO_ANKLE_LEFT: 30,
    SensorType.GYRO_ANKLE_RIGHT: 30,
    SensorType.GYRO_CHEST: 30,
    SensorType.GYRO_TROUSER_FRONT_POCKET: 30,
    SensorType.GYRO_HIP_RIGHT: 30,

    SensorType.ECG_CHEST: 500,

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

    Values:

    - ``sensors``: A dictionary mapping ``SensorType`` to corresponding time series measurements.
    - ``sampling_rate``: Indicates the original sampling rate for each sensor in ``sensors``.
    - ``label``: The activity classification label.
    """
    sensors: Dict[SensorType, np.ndarray]
    sampling_rate: Dict[SensorType, int]
    label: ActivityLabel

    def __post_init__(self):
        """
        Validates the shape of sensor data after initialization.
        Also validates that sensors and sampling_rate have matching dictionary keys.
        """
        for sensor_type, data in self.sensors.items():
            if sensor_type not in SENSOR_AXES_MAP:
                raise ValueError(f"Sensor type {sensor_type} is not supported.")

            expected_axes = SENSOR_AXES_MAP[sensor_type]
            if data.ndim != 2 or data.shape[0] != expected_axes:
                raise ValueError(f"Incorrect shape for sensor type {sensor_type.name}."
                                 f"Expected ({expected_axes}, n_timesteps) but got {data.shape}")

        if set(self.sensors.keys()) != set(self.sampling_rate.keys()):
            raise ValueError(f"Sensor labels and sensor sampling rates do not match keys.")


class HARBaseDataset(torch.utils.data.Dataset, ABC):
    def __init__(
            self,
            dir_name: str,
            train_split_ratio: float,
            validation_split_ratio: float,
            should_resample: bool,
            train_mode: bool,
            validation_mode: bool,
            test_mode: bool,
            window_size: int,
            stride: int,
            root_dir="data",
    ):
        """
        Super class for all datasets. Handles everything from downloading the dataset to loading into ``DataPoint``.

        Uses a sliding window technique to ensure all ``DataPoint`` samples span the same length of time.
        E.g. All sensors measure for 3 seconds.

        :param dir_name: Directory where the dataset should be downloaded.
        :param train_split_ratio: Ratio of training dataset.
        :param validation_split_ratio: Ratio of validation dataset.
        :param should_resample: Whether to resample the dataset to target sampling rate.
        :param train_mode: Dataset's ``__getitem__`` will return training split.
        :param validation_mode: Dataset's ``__getitem__`` will return validation split.
        :param test_mode: Dataset's ``__getitem__`` will return test split.
        :param window_size: Size of the sliding window.
        :param stride: Stride of the sliding window.
        :param root_dir: Root directory where the dataset should be downloaded.
        """
        self.root_dir = root_dir
        self.dir_name = dir_name
        self.train_mode = train_mode
        self.validation_mode = validation_mode
        self.test_mode = test_mode
        self.should_resample = should_resample
        self.train_split_ratio = train_split_ratio
        self.validation_split_ratio = validation_split_ratio


        if sum([train_mode, validation_mode, test_mode]) != 1:
            raise ValueError(f"Only one of train_mode or validation_mode or test_mode allowed."
                             f" Got {[train_mode, validation_mode, test_mode]}.")

        if not os.path.isdir(root_dir):
            try:
                os.mkdir(root_dir)
                print(f"Root directory {root_dir} created.")
            except Exception as e:
                print(f"Failed to create root directory {root_dir}: {e}")

        full_path = os.path.join(root_dir, dir_name)
        is_downloaded = os.path.isdir(full_path)
        if not is_downloaded:
            try:
                os.mkdir(full_path)
                print(f"Directory {full_path} created.")
            except Exception as e:
                print(f"Failed to create {full_path}: {e}")

            self.download_data(root_dir, dir_name, full_path)
        else:
            print(f"Directory {full_path} already exists.")

        long_data_sessions = self.read(root_dir, dir_name, full_path)

        if should_resample:
            self._resample(long_data_sessions)

        self.data = self._create_windows(long_data_sessions, window_size, stride)

        self.size = len(self.data)

    @abstractmethod
    def download_data(self, root_dir, dir_name, full_path):
        pass

    @abstractmethod
    def read(self, root_dir, dir_name, full_path) -> List[DataPoint]:
        """
        Handles loading the files into long, continuous sessions.
        """
        pass

    def _resample(self, long_data_sessions):
        """
        Resample the long sessions according to the target sampling rate. Modifies ``long_data_sessions``
        """
        pass

    def _create_windows(self, long_data_sessions, window_size, stride):
        """
        Uses a sliding window technique to ensure all ``DataPoint`` samples span the same length of time.
        """
        pass

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return self.size



class HARTHDataset(HARBaseDataset):
    def __init__(self, **kwargs):
        kwargs.setdefault("dir_name", 'harth')
        super().__init__(**kwargs)

    def download_data(self, root_dir, dir_name, full_path):
        link = 'https://archive.ics.uci.edu/static/public/779/harth.zip'
        with tempfile.TemporaryDirectory() as tmpdirname:
            print("Downloading HARTH dataset...")
            wget.download(link, tmpdirname)
            zip_file = os.listdir(tmpdirname)[0]
            with zipfile.ZipFile(os.path.join(tmpdirname, zip_file), 'r') as zip_ref:
                zip_ref.extractall(root_dir)
        print(f"Done. Dataset in {full_path}")

    def _get_target_subjects(self, full_path):
        # Each file is a subject. Perform split on subject level
        files = os.listdir(full_path)
        num_subjects = len(files)
        np.random.shuffle(files)

        train_split_index = int(num_subjects * self.train_split_ratio)
        validation_split_index = int(num_subjects * (self.train_split_ratio + self.validation_split_ratio))

        if self.train_mode:
            return files[:train_split_index]
        elif self.validation_mode:
            return files[train_split_index:validation_split_index]
        else:
            return files[validation_split_index:]

    def read(self, root_dir, dir_name, full_path):
        target_subjects = self._get_target_subjects(full_path)

        label_mapping = {
            1: ActivityLabel.WALKING,
            2: ActivityLabel.RUNNING,
            3: ActivityLabel.SHUFFLING,
            4: ActivityLabel.WALKING_UPSTAIR,
            5: ActivityLabel.WALKING_DOWNSTAIR,
            6: ActivityLabel.STANDING,
            7: ActivityLabel.SITTING,
            8: ActivityLabel.LYING,
            13: ActivityLabel.CYCLING, # cycling (sit)
            14: ActivityLabel.CYCLING, # cycling (stand)
            130: ActivityLabel.CYCLING, # cycling (sit, inactive)
            140: ActivityLabel.CYCLING, # cycling (stand, inactive)
        }

        sessions = []

        for subject in target_subjects:
            print(f"Processing {subject}...")
            file_path = os.path.join(full_path, subject)

            df = pd.read_csv(file_path)
            df['label'] = df['label'].map(label_mapping)

            # Create a session ID for consecutive blocks of the same label
            # Prioritizes performance
            df['session_id'] = (df['label'] != df['label'].shift()).cumsum()

            for session_id, session_df in df.groupby('session_id'):
                label = session_df['label'].iloc[0]

                sensors = {
                    SensorType.ACC_BACK_LOWER: session_df[['back_x', 'back_y', 'back_z']].to_numpy().T,
                    SensorType.ACC_THIGH_RIGHT: session_df[['thigh_x', 'thigh_y', 'thigh_z']].to_numpy().T
                }

                sampling_rates = {
                    SensorType.ACC_BACK_LOWER: 50,
                    SensorType.ACC_THIGH_RIGHT: 50,
                }

                sessions.append(DataPoint(sensors=sensors, label=label, sampling_rate=sampling_rates))

        return sessions


if __name__ == "__main__":
    dataset = HARTHDataset()

