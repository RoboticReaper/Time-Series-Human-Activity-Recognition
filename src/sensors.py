from enum import Enum, auto

class SensorFrequency(Enum):
    HIGH = auto()
    LOW = auto()

class Sensor(Enum):
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
    ACC_ARM_LOWER_RIGHT= auto()
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


    EDA = auto()
    HR = auto()
    BVP = auto()
    TEMP_BODY = auto()
    TEMP_SKIN = auto()
    IBI = auto()


# Specifies how many axes each sensor type have
SENSOR_AXES_MAP = {
    Sensor.ACC_BACK_LOWER: 3,
    Sensor.ACC_THIGH_RIGHT: 3,
    Sensor.ACC_WRIST_RIGHT: 3,
    Sensor.ACC_WRIST_LEFT: 3,
    Sensor.ACC_ANKLE_LEFT: 3,
    Sensor.ACC_ANKLE_RIGHT: 3,
    Sensor.ACC_CHEST: 3,
    Sensor.ACC_TROUSER_FRONT_POCKET_RIGHT: 3,
    Sensor.ACC_TROUSER_FRONT_POCKET_LEFT: 3,
    Sensor.ACC_HIP_LEFT: 3,
    Sensor.ACC_HIP_RIGHT: 3,
    Sensor.ACC_ARM_LOWER_RIGHT: 3,
    Sensor.GYRO_WRIST_RIGHT: 3,
    Sensor.GYRO_WRIST_LEFT: 3,
    Sensor.GYRO_ANKLE_LEFT: 3,
    Sensor.GYRO_ANKLE_RIGHT: 3,
    Sensor.GYRO_CHEST: 3,
    Sensor.GYRO_TROUSER_FRONT_POCKET: 3,
    Sensor.GYRO_HIP_RIGHT: 3,
    Sensor.GYRO_ARM_LOWER_RIGHT: 3,
    Sensor.MAGNETOMETER_ANKLE_LEFT: 3,
    Sensor.MAGNETOMETER_ARM_LOWER_RIGHT: 3,

    Sensor.ECG_CHEST: 2,

    Sensor.EDA: 1,
    Sensor.HR: 1,
    Sensor.BVP: 1,
    Sensor.TEMP_BODY: 1,
    Sensor.TEMP_SKIN: 1,
    Sensor.IBI: 1,
}

SENSOR_FREQUENCY_MAP = {
    Sensor.ACC_BACK_LOWER: SensorFrequency.HIGH,
    Sensor.ACC_THIGH_RIGHT: SensorFrequency.HIGH,
    Sensor.ACC_WRIST_RIGHT: SensorFrequency.HIGH,
    Sensor.ACC_WRIST_LEFT: SensorFrequency.HIGH,
    Sensor.ACC_ANKLE_LEFT: SensorFrequency.HIGH,
    Sensor.ACC_ANKLE_RIGHT: SensorFrequency.HIGH,
    Sensor.ACC_CHEST: SensorFrequency.HIGH,
    Sensor.ACC_TROUSER_FRONT_POCKET_RIGHT: SensorFrequency.HIGH,
    Sensor.ACC_TROUSER_FRONT_POCKET_LEFT: SensorFrequency.HIGH,
    Sensor.ACC_HIP_LEFT: SensorFrequency.HIGH,
    Sensor.ACC_HIP_RIGHT: SensorFrequency.HIGH,
    Sensor.ACC_ARM_LOWER_RIGHT: SensorFrequency.HIGH,
    Sensor.GYRO_WRIST_RIGHT: SensorFrequency.HIGH,
    Sensor.GYRO_WRIST_LEFT: SensorFrequency.HIGH,
    Sensor.GYRO_ANKLE_LEFT: SensorFrequency.HIGH,
    Sensor.GYRO_ANKLE_RIGHT: SensorFrequency.HIGH,
    Sensor.GYRO_CHEST: SensorFrequency.HIGH,
    Sensor.GYRO_TROUSER_FRONT_POCKET: SensorFrequency.HIGH,
    Sensor.GYRO_HIP_RIGHT: SensorFrequency.HIGH,
    Sensor.GYRO_ARM_LOWER_RIGHT: SensorFrequency.HIGH,
    Sensor.ECG_CHEST: SensorFrequency.HIGH,
    Sensor.MAGNETOMETER_ANKLE_LEFT: SensorFrequency.HIGH,
    Sensor.MAGNETOMETER_ARM_LOWER_RIGHT: SensorFrequency.HIGH,


    Sensor.EDA: SensorFrequency.LOW,
    Sensor.HR: SensorFrequency.LOW,
    Sensor.BVP: SensorFrequency.LOW,
    Sensor.TEMP_BODY: SensorFrequency.LOW,
    Sensor.TEMP_SKIN: SensorFrequency.LOW,
    Sensor.IBI: SensorFrequency.LOW,
}
