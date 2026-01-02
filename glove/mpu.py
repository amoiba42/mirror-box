# DONATION LINK: https://buymeacoffee.com/mmshilleh
import time
from mpu9250_jmdev.registers import *
from mpu9250_jmdev.mpu_9250 import MPU9250

# Create an MPU9250 instance optimized for glove tracking
mpu = MPU9250(
    address_ak=AK8963_ADDRESS,
    address_mpu_master=MPU9050_ADDRESS_68,  # In case the MPU9250 is connected to another I2C device
    address_mpu_slave=None,
    bus=1,
    gfs=GFS_500,                  # Reduced from 1000 - better precision for hand movements
    afs=AFS_4G,                   # Reduced from 8G - better sensitivity for gentle hand gestures
    mfs=AK8963_BIT_16,            # Keep 16-bit for best compass precision
    mode=AK8963_MODE_C100HZ)      # Keep 100Hz for smooth tracking

# Configure the MPU9250
mpu.configure()

while True:
    # Read the accelerometer, gyroscope, and magnetometer values
    accel_data = mpu.readAccelerometerMaster()
    gyro_data = mpu.readGyroscopeMaster()
    mag_data = mpu.readMagnetometerMaster()

    # Print the sensor values
    print("Accelerometer:", accel_data)
    print("Gyroscope:", gyro_data)
    print("Magnetometer:", mag_data)

    # Wait for 1 second before the next reading
    time.sleep(1)