# How to read motors parameters using the Dynamixel Wizard

## Install the Dynamixel Wizard
You can download the Dynamixel Wizard from the following link:
- [Dynamixel Wizard](https://emanual.robotis.com/docs/en/software/dynamixel/dynamixel_wizard2/)

## Connect to Reachy Mini Lite
1. Power on Reachy Mini Lite using the provided power adapter.
2. Connect your computer to Reachy Mini Lite using a USB-C cable.
3. Open the Dynamixel Wizard application.
4. Click to "option" next to "connect button", then go to the "scan" tab, and check the following info to be able to detect all your motors.
    - Protocol Version: 2.0
    - Baudrate: 1000000
    - Port: Select the appropriate port for your USB connection (e.g., COM3 on Windows or /dev/ttyUSB0 on Linux/Mac).

![Dynamixel_Wizard_Scan_Settings](https://github.com/pollen-robotics/reachy_mini/raw/develop/docs/assets/scan_parameters.png)

5. Click on the "Scan" button to detect all connected motors. The wizard will display a list of detected motors along with their IDs.

## Read motor parameters
1. Select a motor from the list of detected motors.
2. Checks the parameters you want to read (for example: Present Position, Present Velocity, Present Load, etc.)

![Dynamixel_Wizard_Read_Settings](https://github.com/pollen-robotics/reachy_mini/raw/develop/docs/assets/wizard_parameters.png)