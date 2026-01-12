"""Example script to read and display IMU data from Reachy Mini.

This example demonstrates how to access the IMU sensor data (accelerometer,
gyroscope, quaternion orientation, and temperature) from a wireless Reachy Mini.

Note: IMU is only available on the wireless version of Reachy Mini.
"""

import time

from reachy_mini import ReachyMini

with ReachyMini(media_backend="no_media") as mini:
    print("Starting IMU monitoring...")
    print("Press Ctrl+C to stop\n")

    try:
        while True:
            imu_data = mini.imu

            if imu_data is None:
                print("IMU not available (Lite version or no data received yet)")
                time.sleep(1.0)
                continue

            # Extract data
            accel_x, accel_y, accel_z = imu_data["accelerometer"]
            gyro_x, gyro_y, gyro_z = imu_data["gyroscope"]
            quat_w, quat_x, quat_y, quat_z = imu_data["quaternion"]
            temperature = imu_data["temperature"]

            # Display data
            print(
                f"Accelerometer (m/s²): X={accel_x:7.3f}  Y={accel_y:7.3f}  Z={accel_z:7.3f}"
            )
            print(
                f"Gyroscope (rad/s):    X={gyro_x:7.3f}  Y={gyro_y:7.3f}  Z={gyro_z:7.3f}"
            )
            print(
                f"Quaternion (w,x,y,z): W={quat_w:6.3f}  X={quat_x:6.3f}  Y={quat_y:6.3f}  Z={quat_z:6.3f}"
            )
            print(f"Temperature: {temperature:.1f}°C")
            print("-" * 80)

            time.sleep(0.1)  # 10Hz display rate

    except KeyboardInterrupt:
        print("\nStopping IMU monitoring")
