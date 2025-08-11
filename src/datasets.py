import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Tuple, Union
from scipy import signal
from datetime import datetime

from torch.utils.data import ConcatDataset

from sensors import SensorFrequency, Sensor, SENSOR_AXES_MAP
from enum import Enum, auto
from utils import create_conditional_transform

import numpy as np
import torch
import wget
import tempfile
import zipfile
import pandas as pd


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
    WAIST_BEND_FORWARD = auto()
    ARM_RAISE_FORWARD = auto()
    CROUCHING = auto()
    COGNITIVE_TASK = auto()


# Specifies the standard sampling rate for all sensors
TARGET_SAMPLING_RATE = 50


@dataclass
class NumpyDataPoint:
    """
    A dataclass to hold a single sample of sensor data as NumPy arrays, and the label.

    Values:

    - ``sensors``: A dictionary mapping ``SensorType`` to corresponding time series measurements.
    - ``sampling_rate``: Indicates the original sampling rate for each sensor in ``sensors``.
    - ``label``: The activity classification label.
    """
    sensors: Dict[Sensor, np.ndarray]
    sampling_rate: Dict[Sensor, int]
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


@dataclass
class TorchDataPoint(NumpyDataPoint):
    """
    A dataclass to hold sensor data as PyTorch Tensors instead of NumPy arrays.
    """
    sensors: Dict[Sensor, torch.Tensor]
    already_transformed: Dict[Sensor, bool] # to keep track of caching to avoid re-transforming

    def __post_init__(self):
        # Although logic is the same, it's best practice to make the behavior explicit and self-contained
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
    """
        Super class for all HAR datasets. Handles downloading, reading, resampling, and windowing.
        Subclasses must implement ``download_data`` and ``read``, and define class attributes
        for ``_DIR_NAME`` and ``_DOWNLOAD_URL`` and ``_INTERNAL_FOLDER``.
    """
    _DIR_NAME: str # The folder to store the entire dataset
    _DOWNLOAD_URL: str
    _INTERNAL_FOLDER: str # The folder inside the zip after being extracted
    def __init__(
            self,
            train_split_ratio: float,
            validation_split_ratio: float,
            train_mode: bool,
            validation_mode: bool,
            test_mode: bool,
            window_size: int,
            stride: int,
            seed: int,
            transform=None,
            root_dir="data",
    ):
        """
        Super class for all datasets. Handles everything from downloading the dataset to loading into ``DataPoint``.

        Uses a sliding window technique to ensure all ``DataPoint`` samples span the same length of time.
        E.g. All sensors measure for 3 seconds.

        :param train_split_ratio: Ratio of training dataset.
        :param validation_split_ratio: Ratio of validation dataset.
        :param train_mode: Dataset's ``__getitem__`` will return training split.
        :param validation_mode: Dataset's ``__getitem__`` will return validation split.
        :param test_mode: Dataset's ``__getitem__`` will return test split.
        :param window_size: Size of the sliding window.
        :param stride: Stride of the sliding window.
        :param seed: Seed for the random number generator.
        :param transform: Callable function to transform the data while getting item.
        :param root_dir: Root directory where the dataset should be downloaded.
        """
        self.root_dir = root_dir
        self.train_mode = train_mode
        self.validation_mode = validation_mode
        self.test_mode = test_mode
        self.train_split_ratio = train_split_ratio
        self.transform = transform
        self.window_size = window_size
        self.stride = stride
        self.seed = seed
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

        full_path = os.path.join(root_dir, self._DIR_NAME)

        if not os.path.isdir(full_path):
            os.makedirs(full_path, exist_ok=True)
            self.download_data(full_path)

        print(f"Loading {self._DIR_NAME} from {full_path} into memory.")
        long_data_sessions = self.read(full_path)

        print(f"Resampling to {TARGET_SAMPLING_RATE} hertz.")
        self._resample(long_data_sessions)

        print(f"Creating fixed windows.")
        np_data = self._create_windows(long_data_sessions, window_size, stride)

        # Convert numpy datapoint to pytorch datapoint
        self.data = self._np_datapoints_to_torch_datapoints(np_data)

        self.size = len(self.data)

    def _np_datapoints_to_torch_datapoints(self, np_datapoints: List[NumpyDataPoint]) -> List[TorchDataPoint]:
        torch_datapoints = []

        for datapoint in np_datapoints:
            sensors = {
                sensor: torch.from_numpy(data).float()
                for sensor, data in datapoint.sensors.items()
            }

            already_transformed = {sensor: False for sensor in sensors.keys()}

            torch_datapoints.append(TorchDataPoint(sensors=sensors, sampling_rate=datapoint.sampling_rate, label=datapoint.label,
                                                   already_transformed=already_transformed))

        return torch_datapoints

    def _group_by_activity(self, df: pd.DataFrame):
        """Splits a DataFrame into groups of consecutive identical activities."""
        df['session_id'] = (df['label'] != df['label'].shift()).cumsum()
        return df.groupby('session_id')

    def _get_target_subjects(self, subjects_path: str, ignored_files: List[str] = None) -> List[str]:
        """
        Gets a shuffled list of subject file paths for the current split (train/val/test).
        """
        if ignored_files is None:
            ignored_files = []

        subjects = [f for f in os.listdir(subjects_path) if f not in ignored_files]
        rng = np.random.default_rng(self.seed)
        rng.shuffle(subjects)

        num_subjects = len(subjects)
        train_end = int(num_subjects * self.train_split_ratio)
        val_end = int(num_subjects * (self.train_split_ratio + self.validation_split_ratio))

        if self.train_mode:
            selected_subjects = subjects[:train_end]
        elif self.validation_mode:
            selected_subjects = subjects[train_end:val_end]
        else:  # test_mode
            selected_subjects = subjects[val_end:]

        return selected_subjects

    def _resample(self, long_data_sessions: List[NumpyDataPoint]):
        """
        Resample the long sessions according to the target sampling rate. Modifies ``long_data_sessions``
        """
        for dp in long_data_sessions:
            for sensor_type, sensor_data in dp.sensors.items():
                original_rate = dp.sampling_rate[sensor_type]

                if original_rate == TARGET_SAMPLING_RATE:
                    continue

                n_timesteps = sensor_data.shape[1]
                new_timesteps = int(n_timesteps * TARGET_SAMPLING_RATE / original_rate)

                dp.sensors[sensor_type] = signal.resample(x=sensor_data, num=new_timesteps, axis=1)
                dp.sampling_rate[sensor_type] = TARGET_SAMPLING_RATE

    def _create_windows(self, long_data_sessions: List[NumpyDataPoint], window_size, stride) -> List[NumpyDataPoint]:
        """
        Uses a sliding window technique to ensure all ``DataPoint`` samples span the same length of time.
        """
        windowed_data = []
        for session in long_data_sessions:
            first_sensor_type = next(iter(session.sensors))
            session_length = session.sensors[first_sensor_type].shape[1]

            for i in range(0, session_length - window_size + 1, stride):
                windowed_sensors = {st: data[:, i: i + window_size] for st, data in session.sensors.items()}
                if not all(d.shape[1] == window_size for d in windowed_sensors.values()):
                    continue  # Skip windows that aren't full size
                windowed_data.append(NumpyDataPoint(
                    sensors=windowed_sensors,
                    sampling_rate=session.sampling_rate,
                    label=session.label
                ))
        print(f"Created {len(windowed_data)} windowed sessions from {len(long_data_sessions)} sessions.")
        return windowed_data

    def download_data(self, full_path:str):
        """Downloads and extracts the dataset from the `_DOWNLOAD_URL`."""
        with tempfile.TemporaryDirectory() as tmpdirname:
            print(f"Downloading {self._DIR_NAME} dataset from {self._DOWNLOAD_URL}...")
            archive_path = os.path.join(tmpdirname, "dataset.zip")
            wget.download(self._DOWNLOAD_URL, archive_path)
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(full_path)
        print(f"Download complete. Dataset is in {full_path}")

    @abstractmethod
    def read(self, full_path) -> List[NumpyDataPoint]:
        """
        Handles loading the files into long, continuous sessions.

        Check train/validation/test mode and return randomized subject-level splitting based on the splitting ratio.
        """
        pass

    def __getitem__(self, idx) -> TorchDataPoint:
        sample = self.data[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample


    def __len__(self):
        return self.size



class HARTHDataset(HARBaseDataset):
    # https://archive.ics.uci.edu/dataset/779/harth
    _DIR_NAME = 'HARTH'
    _DOWNLOAD_URL = 'https://archive.ics.uci.edu/static/public/779/harth.zip'
    _INTERNAL_FOLDER = 'harth'

    LABEL_MAPPING = {
        1: ActivityLabel.WALKING,
        2: ActivityLabel.RUNNING,
        3: ActivityLabel.SHUFFLING,
        4: ActivityLabel.WALKING_UPSTAIR,
        5: ActivityLabel.WALKING_DOWNSTAIR,
        6: ActivityLabel.STANDING,
        7: ActivityLabel.SITTING,
        8: ActivityLabel.LYING,
        13: ActivityLabel.CYCLING,  # cycling (sit)
        14: ActivityLabel.CYCLING,  # cycling (stand)
        130: ActivityLabel.CYCLING,  # cycling (sit, inactive)
        140: ActivityLabel.CYCLING,  # cycling (stand, inactive)
    }

    SAMPLING_RATES = {
        Sensor.ACC_BACK_LOWER: 50,
        Sensor.ACC_THIGH_RIGHT: 50,
    }

    def read(self, full_path):
        subject_folder = os.path.join(full_path, self._INTERNAL_FOLDER)
        target_subjects = self._get_target_subjects(subject_folder)

        sessions = []

        for subject in target_subjects:
            file_path = os.path.join(subject_folder, subject)
            print(file_path)

            df = pd.read_csv(file_path)
            df['label'] = df['label'].map(self.LABEL_MAPPING)

            for _, session_df in self._group_by_activity(df):
                label = session_df['label'].iloc[0]
                if isinstance(label, ActivityLabel):
                    sensors = {
                        Sensor.ACC_BACK_LOWER: session_df[['back_x', 'back_y', 'back_z']].to_numpy().T,
                        Sensor.ACC_THIGH_RIGHT: session_df[['thigh_x', 'thigh_y', 'thigh_z']].to_numpy().T
                    }

                    sessions.append(NumpyDataPoint(sensors=sensors, label=label, sampling_rate=self.SAMPLING_RATES))

        return sessions

class HAR70Dataset(HARBaseDataset):
    # https://archive.ics.uci.edu/dataset/780/har70
    _DIR_NAME = 'HAR70'
    _DOWNLOAD_URL = 'https://archive.ics.uci.edu/static/public/780/har70.zip'
    _INTERNAL_FOLDER = 'har70plus'

    LABEL_MAPPING = {
        1: ActivityLabel.WALKING,
        3: ActivityLabel.SHUFFLING,
        4: ActivityLabel.WALKING_UPSTAIR,
        5: ActivityLabel.WALKING_DOWNSTAIR,
        6: ActivityLabel.STANDING,
        7: ActivityLabel.SITTING,
        8: ActivityLabel.LYING,
    }

    SAMPLING_RATES = {
        Sensor.ACC_BACK_LOWER: 50,
        Sensor.ACC_THIGH_RIGHT: 50,
    }

    def read(self, full_path):
        subject_folder = os.path.join(full_path, self._INTERNAL_FOLDER)
        target_subjects = self._get_target_subjects(subject_folder)

        sessions = []

        for subject in target_subjects:
            file_path = os.path.join(subject_folder, subject)
            print(file_path)

            df = pd.read_csv(file_path)
            df['label'] = df['label'].map(self.LABEL_MAPPING)

            for _, session_df in self._group_by_activity(df):
                label = session_df['label'].iloc[0]
                if isinstance(label, ActivityLabel):
                    sensors = {
                        Sensor.ACC_BACK_LOWER: session_df[['back_x', 'back_y', 'back_z']].to_numpy().T,
                        Sensor.ACC_THIGH_RIGHT: session_df[['thigh_x', 'thigh_y', 'thigh_z']].to_numpy().T
                    }

                    sessions.append(NumpyDataPoint(sensors=sensors, label=label, sampling_rate=self.SAMPLING_RATES))

        return sessions

class MHEALTHDataset(HARBaseDataset):
    # https://archive.ics.uci.edu/dataset/319/mhealth+dataset
    _DIR_NAME = 'MHEALTH'
    _DOWNLOAD_URL = 'https://archive.ics.uci.edu/static/public/319/mhealth+dataset.zip'
    _INTERNAL_FOLDER = 'MHEALTHDATASET'

    LABEL_MAPPING = {
        1: ActivityLabel.STANDING,
        2: ActivityLabel.SITTING,
        3: ActivityLabel.LYING,
        4: ActivityLabel.WALKING,
        5: ActivityLabel.WALKING_UPSTAIR,
        6: ActivityLabel.WAIST_BEND_FORWARD,
        7: ActivityLabel.ARM_RAISE_FORWARD,
        8: ActivityLabel.CROUCHING,
        9: ActivityLabel.CYCLING,
        10: ActivityLabel.JOGGING,
        11: ActivityLabel.RUNNING,
        12: ActivityLabel.JUMPING,
    }

    SAMPLING_RATES = {
        Sensor.ACC_CHEST: 50,
        Sensor.ECG_CHEST: 50,
        Sensor.ACC_ANKLE_LEFT: 50,
        Sensor.GYRO_ANKLE_LEFT: 50,
        Sensor.MAGNETOMETER_ANKLE_LEFT: 50,
        Sensor.ACC_ARM_LOWER_RIGHT: 50,
        Sensor.GYRO_ARM_LOWER_RIGHT: 50,
        Sensor.MAGNETOMETER_ARM_LOWER_RIGHT: 50
    }

    COLUMN_NAMES = ['acc_chest_x', 'acc_chest_y', 'acc_chest_z', 'ecg_1', 'ecg_2',
                    'acc_ankle_left_x', 'acc_ankle_left_y', 'acc_ankle_left_z',
                    'gyro_ankle_left_x', 'gyro_ankle_left_y', 'gyro_ankle_left_z',
                    'mag_ankle_left_x', 'mag_ankle_left_y', 'mag_ankle_left_z',
                    'acc_right_lower_arm_x', 'acc_right_lower_arm_y', 'acc_right_lower_arm_z',
                    'gyro_right_lower_arm_x', 'gyro_right_lower_arm_y', 'gyro_right_lower_arm_z',
                    'mag_right_lower_arm_x', 'mag_right_lower_arm_y', 'mag_right_lower_arm_z', 'label'
                    ]

    def read(self, full_path) -> List[NumpyDataPoint]:
        subject_folder = os.path.join(full_path, self._INTERNAL_FOLDER)
        target_subjects = self._get_target_subjects(subject_folder, ignored_files=["README.txt"])

        sessions = []

        for subject in target_subjects:
            file_path = os.path.join(subject_folder, subject)
            print(file_path)

            df = pd.read_csv(file_path, sep='\t', header=None, names=self.COLUMN_NAMES)
            df['label'] = df['label'].map(self.LABEL_MAPPING)

            for _, session_df in self._group_by_activity(df):
                label = session_df['label'].iloc[0]
                if isinstance(label, ActivityLabel):
                    sensors = {
                        Sensor.ACC_CHEST: session_df[['acc_chest_x', 'acc_chest_y', 'acc_chest_z']].to_numpy().T,
                        Sensor.ECG_CHEST: session_df[['ecg_1', 'ecg_2']].to_numpy().T,
                        Sensor.ACC_ANKLE_LEFT: session_df[['acc_ankle_left_x', 'acc_ankle_left_y', 'acc_ankle_left_z']].to_numpy().T,
                        Sensor.GYRO_ANKLE_LEFT: session_df[['gyro_ankle_left_x', 'gyro_ankle_left_y', 'gyro_ankle_left_z']].to_numpy().T,
                        Sensor.MAGNETOMETER_ANKLE_LEFT: session_df[['mag_ankle_left_x', 'mag_ankle_left_y', 'mag_ankle_left_z']].to_numpy().T,
                        Sensor.ACC_ARM_LOWER_RIGHT: session_df[['acc_right_lower_arm_x', 'acc_right_lower_arm_y', 'acc_right_lower_arm_z']].to_numpy().T,
                        Sensor.GYRO_ARM_LOWER_RIGHT: session_df[['gyro_right_lower_arm_x', 'gyro_right_lower_arm_y', 'gyro_right_lower_arm_z']].to_numpy().T,
                        Sensor.MAGNETOMETER_ARM_LOWER_RIGHT: session_df[['mag_right_lower_arm_x', 'mag_right_lower_arm_y', 'mag_right_lower_arm_z']].to_numpy().T
                    }

                    sessions.append(NumpyDataPoint(sensors=sensors, label=label, sampling_rate=self.SAMPLING_RATES))

        return sessions


class InducedStressStructuredExerciseWearableDeviceDataset(HARBaseDataset):
    # https://physionet.org/content/wearable-device-dataset/1.0.1/
    _DIR_NAME = 'InducedStressStructuredExerciseWearableDevice'
    _DOWNLOAD_URL = 'https://physionet.org/content/wearable-device-dataset/get-zip/1.0.1/'
    _INTERNAL_FOLDER = "wearable-device-dataset-from-induced-stress-and-structured-exercise-sessions-1.0.1"

    SAMPLING_RATES = {
        Sensor.ACC_WRIST_LEFT: 32,
        Sensor.BVP: 64,
        Sensor.EDA: 4,
        Sensor.HR: 1,
        Sensor.TEMP_SKIN: 4
    }

    def _get_all_subjects_paths(self, full_path) -> List[Tuple[str, str]]:
        """Gets paths for all valid subjects, returning (path, type) tuples."""
        dataset_folder = os.path.join(full_path, self._INTERNAL_FOLDER, "Wearable_Dataset")
        all_subjects = []

        # Define exercise types and subjects to exclude
        exercise_folders = {
            "AEROBIC": ["S11_a", "S11_b"],
            "ANAEROBIC": ["S16_a", "S16_b"],
            "STRESS": ["f14_a", "f14_b", "S02"]
        }

        for exercise_type, exclusions in exercise_folders.items():
            folder_path = os.path.join(dataset_folder, exercise_type)
            subjects = [s for s in os.listdir(folder_path) if s not in exclusions]
            all_subjects.extend([(os.path.join(folder_path, s), exercise_type) for s in subjects])
        return all_subjects

    def _get_target_subjects(self, full_path, ignored_files = None):
        """Overrides base method to handle complex subject structure."""
        all_subjects = self._get_all_subjects_paths(full_path)
        rng = np.random.default_rng(self.seed)
        rng.shuffle(all_subjects)

        num_subjects = len(all_subjects)
        train_end = int(num_subjects * self.train_split_ratio)
        val_end = int(num_subjects * (self.train_split_ratio + self.validation_split_ratio))

        if self.train_mode:
            return all_subjects[:train_end]
        elif self.validation_mode:
            return all_subjects[train_end:val_end]
        else:  # test_mode
            return all_subjects[val_end:]

    def _get_seconds_between_dates(self, start_date_str, end_date_str):
        """
        Calculates the exact number of seconds passed between two dates.

        Args:
            start_date_str (str): The starting date and time in 'yyyy-mm-dd hh:mm:ss' format.
            end_date_str (str): The ending date and time in 'yyyy-mm-dd hh:mm:ss' format.

        Returns:
                int: The total number of seconds between the two dates.
                       Returns a positive value if end_date_str is after start_date_str,
                       a negative value if end_date_str is before start_date_str,
                       and 0 if they are the same.
        """
        date_format = "%Y-%m-%d %H:%M:%S"
        start_dt = datetime.strptime(start_date_str, date_format)
        end_dt = datetime.strptime(end_date_str, date_format)
        time_difference = end_dt - start_dt
        total_seconds = time_difference.total_seconds()

        return int(total_seconds)

    def _get_rows_between_tags(self, data_df, frequency, data_measurement_start_date_str, start_tag=None, end_tag=None):
        start_index = 1  # Skip first row as it tends to be initialization values
        end_index = len(data_df)

        if start_tag is not None:
            start_index = frequency * self._get_seconds_between_dates(data_measurement_start_date_str, start_tag)

        if end_tag is not None:
            end_index = frequency * self._get_seconds_between_dates(data_measurement_start_date_str, end_tag)

        return data_df.iloc[start_index:end_index]

    def _create_data_point(self, data_frames: Dict[Sensor, pd.DataFrame],
                           measurement_start_time: str, label: ActivityLabel,
                           start_tag: str = None, end_tag: str = None) -> NumpyDataPoint:
        """
        Segments all sensor dataframes and creates a DataPoint.
        """
        segmented_sensors = {}
        for sensor_type, df in data_frames.items():
            freq = self.SAMPLING_RATES[sensor_type]
            segmented_df = self._get_rows_between_tags(df, freq, measurement_start_time, start_tag, end_tag)
            segmented_sensors[sensor_type] = segmented_df.to_numpy().T

        return NumpyDataPoint(
            sensors=segmented_sensors,
            sampling_rate=self.SAMPLING_RATES.copy(),
            label=label
        )

    def _process_stress(self, data_frames, start_time, tags, subject_dir_name):
        sessions = []
        protocol_version = subject_dir_name[0]

        if protocol_version == "S":
            tag_times = tags['tag'].tolist()

            # Tuples of (label, start_tag, end_tag)
            segments = [
                (ActivityLabel.RESTING, tag_times[0], tag_times[1]),
                (ActivityLabel.COGNITIVE_TASK, tag_times[2], tag_times[3]),
                (ActivityLabel.RESTING, tag_times[3], tag_times[4]),
                (ActivityLabel.COGNITIVE_TASK, tag_times[4], tag_times[5]),
                (ActivityLabel.RESTING, tag_times[5], tag_times[6]),
                (ActivityLabel.COGNITIVE_TASK, tag_times[6], tag_times[7]),
                (ActivityLabel.COGNITIVE_TASK, tag_times[8], tag_times[9]),
                (ActivityLabel.COGNITIVE_TASK, tag_times[10], tag_times[11]),
            ]
        else:
            # protocol_version is "f"
            # Handle special subject with only 2 sensors
            if subject_dir_name == "f07":
                valid_dataframes = {
                    Sensor.EDA: data_frames[Sensor.EDA],
                    Sensor.ACC_WRIST_LEFT: data_frames[Sensor.ACC_WRIST_LEFT]
                }
                data_frames = valid_dataframes

            tag_times = tags['tag'].tolist()
            segments = [
                (ActivityLabel.RESTING, None, tag_times[0]),
                (ActivityLabel.COGNITIVE_TASK, tag_times[1], tag_times[2]),
                (ActivityLabel.RESTING, tag_times[2], tag_times[3]),
                (ActivityLabel.COGNITIVE_TASK, tag_times[3], tag_times[4]),
                (ActivityLabel.COGNITIVE_TASK, tag_times[5], tag_times[6]),
                (ActivityLabel.RESTING, tag_times[6], tag_times[7]),
                (ActivityLabel.COGNITIVE_TASK, tag_times[7], tag_times[8]),
            ]

        for label, start_tag, end_tag in segments:
            sessions.append(self._create_data_point(data_frames, start_time, label, start_tag, end_tag))

        return sessions

    def _process_aerobic_and_anaerobic(self, data_frames, start_time, tags, subject_dir_name, exercise_type):
        sessions = []
        protocol_version = subject_dir_name[0]

        # Handle special cases with unusual tag structures
        if exercise_type == 'AEROBIC' and subject_dir_name in ['S03', 'S07']:
            tag_end = tags['tag'].iloc[-1] if subject_dir_name == "S03" else tags['tag'].iloc[7]
            sessions.append(self._create_data_point(
                data_frames, start_time,
                label=ActivityLabel.CYCLING, end_tag=tag_end
            ))
            return sessions

        if exercise_type == 'ANAEROBIC' and subject_dir_name == "S06":
            tag_end = tags['tag'].iloc[-1]
            sessions.append(self._create_data_point(
                data_frames, start_time,
                label=ActivityLabel.CYCLING, end_tag=tag_end
            ))
            return sessions

        # Handle normal protocols
        if protocol_version == "S":
            cycle_tag_end = tags['tag'].iloc[-1]
            # Cycling session
            sessions.append(self._create_data_point(
                data_frames, start_time,
                label=ActivityLabel.CYCLING, end_tag=cycle_tag_end
            ))
            # Resting session
            sessions.append(self._create_data_point(
                data_frames, start_time,
                label=ActivityLabel.RESTING, start_tag=cycle_tag_end
            ))
        elif protocol_version == 'f':
            cycle_tag_end_idx = 6 if exercise_type == 'AEROBIC' else -2
            cycle_tag_end = tags['tag'].iloc[cycle_tag_end_idx]
            rest_tag_end = tags['tag'].iloc[-1]

            # Cycling session
            sessions.append(self._create_data_point(
                data_frames, start_time,
                label=ActivityLabel.CYCLING, end_tag=cycle_tag_end
            ))
            # Resting session
            sessions.append(self._create_data_point(
                data_frames, start_time,
                label=ActivityLabel.RESTING, start_tag=cycle_tag_end, end_tag=rest_tag_end
            ))

        return sessions

    def read(self, full_path: str) -> List[NumpyDataPoint]:
        target_subjects = self._get_target_subjects(full_path)
        sessions = []

        for subject_dir, exercise_type in target_subjects:
            print(subject_dir)
            # Load all data files for the subject
            acc_df = pd.read_csv(os.path.join(subject_dir, "ACC.csv"))
            start_time = str(acc_df.columns[0])
            acc_df.columns = ['x', 'y', 'z']

            data_frames = {
                Sensor.ACC_WRIST_LEFT: acc_df,
                Sensor.BVP: pd.read_csv(os.path.join(subject_dir, "BVP.csv"), names=['bvp']),
                Sensor.EDA: pd.read_csv(os.path.join(subject_dir, "EDA.csv"), names=['eda']),
                Sensor.HR: pd.read_csv(os.path.join(subject_dir, "HR.csv"), names=['hr']),
                Sensor.TEMP_SKIN: pd.read_csv(os.path.join(subject_dir, "TEMP.csv"), names=['temp'])
            }
            tags = pd.read_csv(os.path.join(subject_dir, "tags.csv"), header=None, names=['tag'])
            subject_dir_name = os.path.basename(subject_dir)

            # Delegate to the correct processor based on exercise type
            if exercise_type == "STRESS":
                sessions.extend(self._process_stress(data_frames, start_time, tags, subject_dir_name))
            else:
                sessions.extend(self._process_aerobic_and_anaerobic(data_frames, start_time, tags, subject_dir_name, exercise_type))

        return sessions


def get_full_transformed_dataset(train_split_ratio, validation_split_ratio, window_size, stride, seed, n_fft, hop_length, win_length, train_mode = True, validation_mode = False, test_mode = False, ):
    dataset1 = HARTHDataset(train_mode=train_mode, train_split_ratio=train_split_ratio,
                            validation_split_ratio=validation_split_ratio,
                            validation_mode=validation_mode, test_mode=test_mode, window_size=window_size, stride=stride, seed=seed)
    dataset2 = MHEALTHDataset(train_mode=train_mode, train_split_ratio=train_split_ratio,
                              validation_split_ratio=validation_split_ratio,
                              validation_mode=validation_mode, test_mode=test_mode, window_size=window_size, stride=stride, seed=seed)
    dataset3 = InducedStressStructuredExerciseWearableDeviceDataset(train_mode=train_mode,
                                                                    train_split_ratio=train_split_ratio,
                                                                    validation_split_ratio=validation_split_ratio,
                                                                    validation_mode=validation_mode, test_mode=test_mode,
                                                                    window_size=window_size, stride=stride, seed=seed)
    dataset4 = HAR70Dataset(train_mode=train_mode, train_split_ratio=train_split_ratio,
                            validation_split_ratio=validation_split_ratio,
                            validation_mode=validation_mode, test_mode=test_mode, window_size=window_size, stride=stride, seed=seed)

    all_datasets = [dataset1, dataset2, dataset3, dataset4]
    concat_dataset_for_transforms = ConcatDataset(all_datasets)

    # obtain the transform function for each dataset
    transform = create_conditional_transform(concat_dataset_for_transforms, n_fft, hop_length, win_length)

    for d in all_datasets:
        d.transform = transform

    return ConcatDataset(all_datasets)

if __name__ == "__main__":
    window_size = 250
    stride = 100
    seed = 42
    train_split_ratio = 0.8
    validation_split_ratio = 0.1
    n_fft = 50
    hop_length = 25
    win_length = 50

    dataset = get_full_transformed_dataset(train_split_ratio, validation_split_ratio, window_size, stride, seed, n_fft, hop_length, win_length)
    import time
    start = time.perf_counter()
    a = dataset[2000]
    end = time.perf_counter()
    print(end - start)
    print((end - start) * len(dataset))
    start = time.perf_counter()
    b = dataset[2000]
    end = time.perf_counter()
    print(end - start)
    print((end - start) * len(dataset))
