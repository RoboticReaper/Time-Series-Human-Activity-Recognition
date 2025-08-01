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
    def __init__(self, item_id, frequency):
        self._value_ = item_id
        self.frequency = frequency

    ACC_BACK_LOWER = (1, SensorFrequency.HIGH)
    ACC_THIGH_RIGHT = (2, SensorFrequency.HIGH)
    ACC_WRIST_RIGHT = (3, SensorFrequency.HIGH)
    ACC_WRIST_LEFT = (4, SensorFrequency.HIGH)
    ACC_ANKLE_LEFT = (5, SensorFrequency.HIGH)
    ACC_ANKLE_RIGHT = (6, SensorFrequency.HIGH)
    ACC_CHEST = (7, SensorFrequency.HIGH)
    ACC_TROUSER_FRONT_POCKET_RIGHT = (8, SensorFrequency.HIGH)
    ACC_TROUSER_FRONT_POCKET_LEFT = (9, SensorFrequency.HIGH)
    ACC_HIP_LEFT = (10, SensorFrequency.HIGH)
    ACC_HIP_RIGHT = (11, SensorFrequency.HIGH)
    ACC_ARM_LOWER_RIGHT= (12, SensorFrequency.HIGH)
    GYRO_WRIST_RIGHT = (19, SensorFrequency.HIGH)
    GYRO_WRIST_LEFT = (20, SensorFrequency.HIGH)
    GYRO_ANKLE_LEFT = (21, SensorFrequency.HIGH)
    GYRO_ANKLE_RIGHT = (22, SensorFrequency.HIGH)
    GYRO_CHEST = (23, SensorFrequency.HIGH)
    GYRO_TROUSER_FRONT_POCKET = (24, SensorFrequency.HIGH)
    GYRO_HIP_RIGHT = (25, SensorFrequency.HIGH)
    GYRO_ARM_LOWER_RIGHT = (26, SensorFrequency.HIGH)
    ECG_CHEST = (27, SensorFrequency.HIGH)
    MAGNETOMETER_ANKLE_LEFT = (28, SensorFrequency.HIGH)
    MAGNETOMETER_ARM_LOWER_RIGHT = (29, SensorFrequency.HIGH)


    EDA = (13, SensorFrequency.LOW)
    HR = (14, SensorFrequency.LOW)
    BVP = (15, SensorFrequency.LOW)
    TEMP_BODY = (16, SensorFrequency.LOW)
    TEMP_SKIN = (17, SensorFrequency.LOW)
    IBI = (18, SensorFrequency.LOW)


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
