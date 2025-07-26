import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Dict
from scipy import signal
from datetime import datetime

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
    ACC_ARM_LOWER_RIGHT= auto(),
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
    GYRO_ARM_LOWER_RIGHT = auto()
    ECG_CHEST = auto()
    MAGNETOMETER_ANKLE_LEFT = auto()
    MAGNETOMETER_ARM_LOWER_RIGHT = auto()


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
    SensorType.ACC_ARM_LOWER_RIGHT: 3,
    SensorType.GYRO_WRIST_RIGHT: 3,
    SensorType.GYRO_WRIST_LEFT: 3,
    SensorType.GYRO_ANKLE_LEFT: 3,
    SensorType.GYRO_ANKLE_RIGHT: 3,
    SensorType.GYRO_CHEST: 3,
    SensorType.GYRO_TROUSER_FRONT_POCKET: 3,
    SensorType.GYRO_HIP_RIGHT: 3,
    SensorType.GYRO_ARM_LOWER_RIGHT: 3,
    SensorType.MAGNETOMETER_ANKLE_LEFT: 3,
    SensorType.MAGNETOMETER_ARM_LOWER_RIGHT: 3,

    SensorType.ECG_CHEST: 2,

    SensorType.EDA: 1,
    SensorType.HR: 1,
    SensorType.BVP: 1,
    SensorType.TEMP_BODY: 1,
    SensorType.TEMP_SKIN: 1,
    SensorType.IBI: 1,
}


# Specifies the standard sampling rate for all sensors
TARGET_SAMPLING_RATE = 50


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

        :param dir_name: Directory where the dataset should be downloaded.
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
        self.dir_name = dir_name
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

        full_path = os.path.join(root_dir, dir_name)
        is_downloaded = os.path.isdir(full_path)
        if not is_downloaded:
            try:
                os.mkdir(full_path)
                print(f"Directory {full_path} created.")
            except Exception as e:
                print(f"Failed to create {full_path}: {e}")

            self.download_data(root_dir, dir_name, full_path)

        print(f"Loading {full_path} into memory.")
        long_data_sessions = self.read(root_dir, dir_name, full_path)

        print(f"Resampling to {TARGET_SAMPLING_RATE} hertz.")
        self._resample(long_data_sessions)

        print(f"Creating fixed windows.")
        self.data = self._create_windows(long_data_sessions, window_size, stride)

        self.size = len(self.data)

    def _resample(self, long_data_sessions: List[DataPoint]):
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

    def _create_windows(self, long_data_sessions: List[DataPoint], window_size, stride):
        """
        Uses a sliding window technique to ensure all ``DataPoint`` samples span the same length of time.
        """
        windowed_data = []
        for session in long_data_sessions:
            first_sensor_type = next(iter(session.sensors))
            session_length = session.sensors[first_sensor_type].shape[1]

            for i in range(0, session_length - window_size + 1, stride):
                windowed_sensors = {st: data[:, i: i + window_size] for st, data in session.sensors.items()}
                windowed_data.append(DataPoint(
                    sensors=windowed_sensors,
                    sampling_rate=session.sampling_rate,
                    label=session.label
                ))
        print(f"Created {len(windowed_data)} windowed sessions from {len(long_data_sessions)} sessions.")
        return windowed_data

    @abstractmethod
    def download_data(self, root_dir, dir_name, full_path):
        pass

    @abstractmethod
    def read(self, root_dir, dir_name, full_path) -> List[DataPoint]:
        """
        Handles loading the files into long, continuous sessions.

        Check train/validation/test mode and return randomized subject-level splitting based on the splitting ratio.
        """
        pass

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return self.size



class HARTHDataset(HARBaseDataset):
    # https://archive.ics.uci.edu/dataset/779/harth
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
        SensorType.ACC_BACK_LOWER: 50,
        SensorType.ACC_THIGH_RIGHT: 50,
    }

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
        rng = np.random.default_rng(self.seed)
        rng.shuffle(files)

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

        sessions = []

        for subject in target_subjects:
            file_path = os.path.join(full_path, subject)

            df = pd.read_csv(file_path)
            df['label'] = df['label'].map(self.LABEL_MAPPING)

            # Create a session ID for consecutive blocks of the same label
            # Prioritizes performance
            df['session_id'] = (df['label'] != df['label'].shift()).cumsum()

            for session_id, session_df in df.groupby('session_id'):
                label = session_df['label'].iloc[0]
                if isinstance(label, ActivityLabel):
                    sensors = {
                        SensorType.ACC_BACK_LOWER: session_df[['back_x', 'back_y', 'back_z']].to_numpy().T,
                        SensorType.ACC_THIGH_RIGHT: session_df[['thigh_x', 'thigh_y', 'thigh_z']].to_numpy().T
                    }

                    sessions.append(DataPoint(sensors=sensors, label=label, sampling_rate=self.SAMPLING_RATES))

        return sessions

class MHEALTHDataset(HARBaseDataset):
    # https://archive.ics.uci.edu/dataset/319/mhealth+dataset
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
        SensorType.ACC_CHEST: 50,
        SensorType.ECG_CHEST: 50,
        SensorType.ACC_ANKLE_LEFT: 50,
        SensorType.GYRO_ANKLE_LEFT: 50,
        SensorType.MAGNETOMETER_ANKLE_LEFT: 50,
        SensorType.ACC_ARM_LOWER_RIGHT: 50,
        SensorType.GYRO_ARM_LOWER_RIGHT: 50,
        SensorType.MAGNETOMETER_ARM_LOWER_RIGHT: 50
    }

    COLUMN_NAMES = ['acc_chest_x', 'acc_chest_y', 'acc_chest_z', 'ecg_1', 'ecg_2',
                    'acc_ankle_left_x', 'acc_ankle_left_y', 'acc_ankle_left_z',
                    'gyro_ankle_left_x', 'gyro_ankle_left_y', 'gyro_ankle_left_z',
                    'mag_ankle_left_x', 'mag_ankle_left_y', 'mag_ankle_left_z',
                    'acc_right_lower_arm_x', 'acc_right_lower_arm_y', 'acc_right_lower_arm_z',
                    'gyro_right_lower_arm_x', 'gyro_right_lower_arm_y', 'gyro_right_lower_arm_z',
                    'mag_right_lower_arm_x', 'mag_right_lower_arm_y', 'mag_right_lower_arm_z', 'label'
                    ]

    def __init__(self, **kwargs):
        kwargs.setdefault("dir_name", 'MHEALTHDATASET')
        super().__init__(**kwargs)

    def download_data(self, root_dir, dir_name, full_path):
        link = 'https://archive.ics.uci.edu/static/public/319/mhealth+dataset.zip'
        with tempfile.TemporaryDirectory() as tmpdirname:
            print("Downloading MHEALTH dataset...")
            wget.download(link, tmpdirname)
            zip_file = os.listdir(tmpdirname)[0]
            with zipfile.ZipFile(os.path.join(tmpdirname, zip_file), 'r') as zip_ref:
                zip_ref.extractall(root_dir)
        print(f"Done. Dataset in {full_path}")

    def _get_target_subjects(self, full_path):
        # Each file is a subject, except README.txt. Perform subject-level splitting
        files = os.listdir(full_path)
        files.remove("README.txt")
        num_subjects = len(files)
        rng = np.random.default_rng(self.seed)
        rng.shuffle(files)

        train_split_index = int(num_subjects * self.train_split_ratio)
        validation_split_index = int(num_subjects * (self.train_split_ratio + self.validation_split_ratio))

        if self.train_mode:
            return files[:train_split_index]
        elif self.validation_mode:
            return files[train_split_index:validation_split_index]
        else:
            return files[validation_split_index:]

    def read(self, root_dir, dir_name, full_path) -> List[DataPoint]:
        target_subjects = self._get_target_subjects(full_path)

        sessions = []

        for subject in target_subjects:
            file_path = os.path.join(full_path, subject)

            df = pd.read_csv(file_path, sep='\t', header=None, names=self.COLUMN_NAMES)
            df['label'] = df['label'].map(self.LABEL_MAPPING)

            df['session_id'] = (df['label'] != df['label'].shift()).cumsum()

            for session_id, session_df in df.groupby('session_id'):
                label = session_df['label'].iloc[0]
                if isinstance(label, ActivityLabel):
                    sensors = {
                        SensorType.ACC_CHEST: session_df[['acc_chest_x', 'acc_chest_y', 'acc_chest_z']].to_numpy().T,
                        SensorType.ECG_CHEST: session_df[['ecg_1', 'ecg_2']].to_numpy().T,
                        SensorType.ACC_ANKLE_LEFT: session_df[['acc_ankle_left_x', 'acc_ankle_left_y', 'acc_ankle_left_z']].to_numpy().T,
                        SensorType.GYRO_ANKLE_LEFT: session_df[['gyro_ankle_left_x', 'gyro_ankle_left_y', 'gyro_ankle_left_z']].to_numpy().T,
                        SensorType.MAGNETOMETER_ANKLE_LEFT: session_df[['mag_ankle_left_x', 'mag_ankle_left_y', 'mag_ankle_left_z']].to_numpy().T,
                        SensorType.ACC_ARM_LOWER_RIGHT: session_df[['acc_right_lower_arm_x', 'acc_right_lower_arm_y', 'acc_right_lower_arm_z']].to_numpy().T,
                        SensorType.GYRO_ARM_LOWER_RIGHT: session_df[['gyro_right_lower_arm_x', 'gyro_right_lower_arm_y', 'gyro_right_lower_arm_z']].to_numpy().T,
                        SensorType.MAGNETOMETER_ARM_LOWER_RIGHT: session_df[['mag_right_lower_arm_x', 'mag_right_lower_arm_y', 'mag_right_lower_arm_z']].to_numpy().T
                    }

                    sessions.append(DataPoint(sensors=sensors, label=label, sampling_rate=self.SAMPLING_RATES))

        return sessions


class InducedStressStructuredExerciseWearableDeviceDataset(HARBaseDataset):
    # https://physionet.org/content/wearable-device-dataset/1.0.1/
    inside_folder_name = "wearable-device-dataset-from-induced-stress-and-structured-exercise-sessions-1.0.1"

    def __init__(self, **kwargs):
        kwargs.setdefault("dir_name", 'InducedStressStructuredExerciseWearableDevice')
        super().__init__(**kwargs)

    def download_data(self, root_dir, dir_name, full_path):
        link = 'https://physionet.org/content/wearable-device-dataset/get-zip/1.0.1/'
        with tempfile.TemporaryDirectory() as tmpdirname:
            print("Downloading MHEALTH dataset...")
            wget.download(link, tmpdirname)
            zip_file = os.listdir(tmpdirname)[0]
            with zipfile.ZipFile(os.path.join(tmpdirname, zip_file), 'r') as zip_ref:
                zip_ref.extractall(full_path)
        print(f"Done. Dataset in {full_path}")

    def _get_target_subjects(self, full_path):
        subjects = []
        dataset_folder = os.path.join(full_path, self.inside_folder_name, "Wearable_Dataset")
        aerobic_folder = os.path.join(dataset_folder, "AEROBIC")
        aerobic_subjects = os.listdir(aerobic_folder)

        # prevent leaking subject patterns between train/valid/test
        aerobic_subjects.remove("S11_a")
        aerobic_subjects.remove("S11_b")

        subjects.extend([os.path.join(aerobic_folder, subject) for subject in aerobic_subjects])

        anaerobic_folder = os.path.join(dataset_folder, "ANAEROBIC")
        anaerobic_subjects = os.listdir(anaerobic_folder)
        anaerobic_subjects.remove("S16_a")
        anaerobic_subjects.remove("S16_b")

        subjects.extend([os.path.join(anaerobic_folder, subject) for subject in anaerobic_subjects])

        stress_folder = os.path.join(dataset_folder, "STRESS")
        stress_subjects = os.listdir(stress_folder)
        stress_subjects.remove("f14_a")
        stress_subjects.remove("f14_b")
        stress_subjects.remove("S02")

        subjects.extend([os.path.join(stress_folder, subject) for subject in stress_subjects])

        num_subjects = len(subjects)
        rng = np.random.default_rng(self.seed)
        rng.shuffle(subjects)

        train_split_index = int(num_subjects * self.train_split_ratio)
        validation_split_index = int(num_subjects * (self.train_split_ratio + self.validation_split_ratio))

        if self.train_mode:
            return subjects[:train_split_index]
        elif self.validation_mode:
            return subjects[train_split_index:validation_split_index]
        else:
            return subjects[validation_split_index:]

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
        if start_tag is None and end_tag is None:
            raise ValueError("Either start_tag or end_tag must be provided.")

        start_index = 1  # Default start index, skips potential initialization values
        end_index = len(data_df)

        if start_tag is not None:
            start_index = frequency * self._get_seconds_between_dates(data_measurement_start_date_str, start_tag)

        if end_tag is not None:
            end_index = frequency * self._get_seconds_between_dates(data_measurement_start_date_str, end_tag)

        return data_df.iloc[start_index:end_index]

    def _create_data_point_from_segment(self, data_frames: Dict[SensorType, pd.DataFrame],
                                        sampling_rates: Dict[SensorType, int],
                                        measurement_start_time: str, label: ActivityLabel,
                                        start_tag: str = None, end_tag: str = None) -> DataPoint:
        """
        Segments all sensor dataframes and creates a DataPoint.
        """
        segmented_sensors = {}
        for sensor_type, df in data_frames.items():
            freq = sampling_rates[sensor_type]
            segmented_df = self._get_rows_between_tags(df, freq, measurement_start_time, start_tag, end_tag)
            segmented_sensors[sensor_type] = segmented_df.to_numpy().T

        return DataPoint(
            sensors=segmented_sensors,
            sampling_rate=sampling_rates,
            label=label
        )

    def read(self, root_dir, dir_name, full_path) -> List[DataPoint]:
        target_subjects_dir = self._get_target_subjects(full_path)
        sessions = []

        sampling_rates = {
            SensorType.ACC_WRIST_LEFT: 32,
            SensorType.BVP: 64,
            SensorType.EDA: 4,
            SensorType.HR: 1,
            SensorType.TEMP_SKIN: 4
        }

        for subject_dir in target_subjects_dir:
            parent_dir_path = os.path.dirname(subject_dir)
            subject_exercise_type = os.path.basename(parent_dir_path)
            subject_dir_name = os.path.basename(subject_dir)
            subject_protocol_version = subject_dir_name[0]

            # Load data into a dictionary of dataframes
            acc_df = pd.read_csv(os.path.join(subject_dir, "ACC.csv"))
            measurement_start_time = str(acc_df.columns[0])
            acc_df.columns = ['x', 'y', 'z']

            data_frames = {
                SensorType.ACC_WRIST_LEFT: acc_df,
                SensorType.BVP: pd.read_csv(os.path.join(subject_dir, "BVP.csv"), names=['bvp']),
                SensorType.EDA: pd.read_csv(os.path.join(subject_dir, "EDA.csv"), names=['eda']),
                SensorType.HR: pd.read_csv(os.path.join(subject_dir, "HR.csv"), names=['hr']),
                SensorType.TEMP_SKIN: pd.read_csv(os.path.join(subject_dir, "TEMP.csv"), names=['temp'])
            }
            tags = pd.read_csv(os.path.join(subject_dir, "tags.csv"), header=None, names=['tag'])

            if subject_exercise_type == "AEROBIC":
                if subject_protocol_version == "S":
                    if subject_dir_name in ["S03", "S07"]:
                        tag_end = tags['tag'].iloc[-1] if subject_dir_name == "S03" else tags['tag'].iloc[7]
                        sessions.append(self._create_data_point_from_segment(
                            data_frames, sampling_rates, measurement_start_time,
                            label=ActivityLabel.CYCLING, end_tag=tag_end
                        ))
                    else:
                        cycle_tag_end = tags['tag'].iloc[-1]
                        # Cycling session
                        sessions.append(self._create_data_point_from_segment(
                            data_frames, sampling_rates, measurement_start_time,
                            label=ActivityLabel.CYCLING, end_tag=cycle_tag_end
                        ))
                        # Resting session
                        sessions.append(self._create_data_point_from_segment(
                            data_frames, sampling_rates, measurement_start_time,
                            label=ActivityLabel.RESTING, start_tag=cycle_tag_end
                        ))

                elif subject_protocol_version == "f":
                    cycle_tag_end = tags['tag'].iloc[6]
                    rest_tag_end = tags['tag'].iloc[-1]
                    # Cycling session
                    sessions.append(self._create_data_point_from_segment(
                        data_frames, sampling_rates, measurement_start_time,
                        label=ActivityLabel.CYCLING, end_tag=cycle_tag_end
                    ))
                    # Resting session
                    sessions.append(self._create_data_point_from_segment(
                        data_frames, sampling_rates, measurement_start_time,
                        label=ActivityLabel.RESTING, start_tag=cycle_tag_end, end_tag=rest_tag_end
                    ))

            elif subject_exercise_type == "ANAEROBIC":
                pass

        return sessions


if __name__ == "__main__":
    # dataset1 = HARTHDataset(train_mode=True, train_split_ratio=0.8, validation_split_ratio=0.1,
    #                        validation_mode=False, test_mode=False, window_size=60, stride=60, seed=42)
    # dataset2 = MHEALTHDataset(train_mode=True, train_split_ratio=0.8, validation_split_ratio=0.1,
    #                        validation_mode=False, test_mode=False, window_size=60, stride=60, seed=42)
    dataset3 = InducedStressStructuredExerciseWearableDeviceDataset(train_mode=True, train_split_ratio=0.8, validation_split_ratio=0.1,
                              validation_mode=False, test_mode=False, window_size=60, stride=60, seed=42)

