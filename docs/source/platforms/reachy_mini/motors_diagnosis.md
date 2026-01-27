# Motors diagnosis and troubleshooting guide

This document provides a guide to diagnose and troubleshoot common issues related to the motors of the Reachy Mini robot.

## Reachy Mini Testbench app
To help diagnose motor issues, we have developed the [Reachy Mini Testbench app](https://huggingface.co/spaces/pollen-robotics/reachy_mini_testbench). This app allows you to test individual motors, check their status, and identify potential problems.
You need to install this app on the robot before using it. If you have a lite version, you will need to run the daemon without autostart in a terminal and access the dashboard, as explained below:
<details>
<summary>Instructions for Lite version</summary>

- _Virtual Environment:_ Ensure you are running inside your virtual environment (`.venv`, `reachy_mini_env`,...).
- _SDK Update:_ Ensure you have the latest version.
    With `pip`, run :
    ```bash
    pip install -U reachy-mini
    ```
    With `uv`, run :
    ```bash
    uv pip install -U reachy-mini
    ```

- _Daemon:_ Run the daemon `reachy-mini-daemon --no-autostart`
- _Access Dashboard:_ Open your web browser and go to `http://localhost:8000/`.  
There, you can find the Testbench app in the "Apps" section, install it then run it.


</details>

## Motors typical troubleshooting process
If you have any of the following symptoms, please follow the diagnosis steps below.
- Motors blinking red and not responding/moving.
- No error but motors don't respond to commands.
- Motors showing errors like "Overload Error"
- Missing motors: "No motor found on port", "Missing motor", etc.

### Diagnosis
- Power on the robot.
- Open the Reachy Mini Testbench app.
- Do a motor scan by clicking on the "Scan Motors" button.

### Possible outcomes

**1. All motors are detected:**  
Then all the motors are physically connected and working.

![All_motors_detected](https://github.com/pollen-robotics/reachy_mini/raw/develop/docs/assets/all_motors_found.png) 

If you still have issues, they are maybe not in the right order or with the right configuration.
Please click the "Check all motors" to verify all motor configurations match expected values.

![Check_all_motors](https://github.com/pollen-robotics/reachy_mini/raw/develop/docs/assets/check_motor_configuration.png)

=> If anything is not correct, the daemon will reflash all the motors when you restart it.
If everything is correct, but you still have issues, please refer to the verifications below.

**Motors inversion:** If you get "Motor hardware errors: ['Overload Error']" and blinking leds a few seconds after starting the robot **for the first time** and have two motors arm pointing upward.  
It is VERY likely there are motors not placed in the correct slot, e.g. motor 1 on slot 2.

![Motors inversion symptom](https://github.com/pollen-robotics/reachy_mini/raw/develop/docs/assets/motors_upward.png)

**Check the arm orientation on the motor's horn**:
Remove the faulty motor, then place the arm upward like in the attached picture. Then check if you can see the two line marks aligned as represented:

![Marks_aligned](https://github.com/pollen-robotics/reachy_mini/raw/develop/docs/assets/marks_aligned.png)

If they are not, please remove the two screws securing the arm and put it back with the two lines matching.  


**Check the extra length of the usb cable inside the head:**  
If it's too long inside the head, there might be insufficient slack underneath and the head cannot move freely.  
So the motors force too much and can be damaged.

![usb_cable_length](https://github.com/pollen-robotics/reachy_mini/raw/develop/docs/assets/usb_cable_length.jpg)

Please let some slack to the usb cable to allow the head to move freely, even to its maximal height position.  



**A motor feels broken:**
We identified an issue affecting a limited production batch of Reachy Mini robots, related to a faulty batch of Dynamixel motors. 

In most reported cases, the issue affects motor number 4 or one with QC label n°2544.

If one of your motors, **blinks red and feels unusually hard to move when turned off** [(example video here)](https://drive.google.com/file/d/1UHTqUcb21aFThqlr2Qcx23VWqvj_y-ly/view?usp=sharing), and you are 100% sure the motor was in the correct slot.

It's probably a broken motor.

First, try to update your robot to the latest software version, then reboot it. This will reflash your motors.
If the issue persists, please contact support.

**2. Some motors are not detected:**.

![Some_motors_missing](https://github.com/pollen-robotics/reachy_mini/raw/develop/docs/assets/missing_motors.png)

- Only one motor is missing: Check the physical connection of that motor according to the assembly instructions.  
If you are 100% sure the connection is okay, it can also be a motor not flashed properly. Please refer to the **Flashing issue** section below.

- If motors with successive IDs are missing (e.g., motors 1-2-3, or 4-5-6, or 17-18 are missing), check the physical connections between the power board in the foot and the motors 3 and 4. If the problem occurs on 17-18, check the connexion between "R" motor and the pcb board in the head.

- If **two motors are missing** but the connection is okay, then maybe you got two same motors ID in your kit. Three is very unlikely. Check the "Flashing issue" section below.

### Flashing issue
If one or more motors are not detected, it can be due to a flashing issue affecting the motor IDs or baudrate. 

To solve this, you will need:  
1. Click the "Scan all baudrates" button in the Testbench app.
The motors must be detected at baudrate 1000000.
The motors IDs must be from 10 to 18 without duplicates or missing numbers.

2. If there is a motor with a wrong baudrate or/and ID, you will need to reflash it with the correct parameters.
To do so, please follow the flashing procedure below:
- In the motor reflash section, choose the motor with the wrong ID or baudrate.
- Choose the preset of what the motor should be (e.g., motor 10 for the base rotation).
- Click on "Reflash motor".

![Reflash_motor](https://github.com/pollen-robotics/reachy_mini/raw/develop/docs/assets/reflash_motors.png)

3. If you have multiple motors missing and are sure the connections are okay, you will need to reflash all the faulty/missing motors one by one, following the same procedure as above but unplugging all the other faulty motors to avoid ID conflicts.
