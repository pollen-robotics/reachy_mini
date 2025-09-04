# Reachy Mini API Documentation

*⚠️ All examples shown below suppose that you have already started the Reachy Mini daemon, either by running `reachy-mini-daemon` or by using the Python module `reachy_mini.daemon.cli`. ⚠️*

## ReachyMini

Reachy Mini's API is designed to be simple and intuitive. You will mostly interact with the `ReachyMini` class, which provides methods to control the robot's joints such as the head and antennas and interact with its sensors.

The first step is to instantiate the `ReachyMini` class. This can be done as follows:

```python
from reachy_mini import ReachyMini

mini = ReachyMini()
```

This will connect to the Reachy Mini daemon, which is responsible for managing the hardware communication with the robot's motors and sensors. As soon as the `ReachyMini` instance is created, it will automatically connect to the daemon and initialize the robot's components. 

To ensure that the connection is properly established and cleaned up, it is recommended to use the `with` statement when working with the `ReachyMini` class. This will automatically handle the connection and disconnection of the daemon:

```python
from reachy_mini import ReachyMini

with ReachyMini() as mini:
    # Your code here
```

### Moving the robot

Then, the next step is to show how to move the robot. The `ReachyMini` class provides methods called `set_target` and `goto_target` that allows you to move the robot's joints to a specific target position. You can control:
* the head's position and orientation
* the body's rotation angle
* the antennas' position

For instance, to move the head of the robot slightly to the left then go back, you can use the following code:

```python
from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose

with ReachyMini() as reachy:
    # Move the head to a specific position
    reachy.goto_target(head=create_head_pose(y=-10, mm=True))

    # Goes back to the initial position
    reachy.goto_target(head=create_head_pose(y=0, mm=True))
```

Let's break down the code:

* First, `create_head_pose(y=-10, mm=True)` creates a pose for the head where the y-axis is set to -10 mm. The `mm=True` argument indicates that the value is in millimeters. 
The pose is a 4x4 transformation matrix that defines the position and orientation of the head. You can print the pose to see its values.
```python
print(create_head_pose(y=-10, mm=True))
>>>
[[ 1.   0.   0.   0.  ]
 [ 0.   1.   0.  -0.01]
 [ 0.   0.   1.   0.  ]
 [ 0.   0.   0.   1.  ]]
```

* Then, this matrix is passed to the `goto_target` method, as the head target.

The `goto_target` method accept more arguments, such as `duration` to specify how long the movement should take. By default, its duration is set to 0.5 seconds, but you can change it to any value you want:

```python
reachy.goto_target(head=create_head_pose(y=-10, mm=True), duration=2.0)
```

You can also orient the head by passing additional arguments to the `create_head_pose` function. For example, to roll the head 15 degrees while moving it up 10 mm, you can do:

```python
from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose

with ReachyMini() as reachy:
    # Move the head up (10mm on z-axis) and roll it 15 degrees
    pose = create_head_pose(z=10, roll=15, degrees=True, mm=True)
    reachy.goto_target(head=pose, duration=2.0)

    # Reset to default pose
    pose = create_head_pose() 
    reachy.goto_target(head=pose, duration=2.0)
```


You can also make the antennas move by passing the `antennas` argument to the `goto_target` method. For example, to move the antennas to a specific position:

```python
import numpy as np

from reachy_mini import ReachyMini

with ReachyMini() as reachy:
    # Move the antennas to a specific position
    reachy.goto_target(antennas=np.deg2rad([45, 45]), duration=1.0)

    # Reset the antennas to their initial position
    reachy.goto_target(antennas=[0, 0], duration=1.0)
```

You need to pass the angles in radians, so you can use `numpy.deg2rad` to convert degrees to radians. The first value in the list corresponds to the left antenna, and the second value corresponds to the right antenna.

You can also move the head, the body and the antennas at the same time by passing all three arguments to the `goto_target` method:

```python
import numpy as np

from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose

with ReachyMini() as reachy:
    # Move the head and antennas to a specific position
    reachy.goto_target(
        head=create_head_pose(y=-10, mm=True),
        antennas=np.deg2rad([45, 45]),
        duration=2.0,
        body_yaw=np.deg2rad(30)
    )
```

Finally, you can also select the interpolation method used. By default, the interpolation method is set to "minjerk", but you can change it to "linear", "cartoon" or "ease".

```python
import numpy as np

from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose

with ReachyMini() as reachy:
    # Move the head and antennas to a specific position
    reachy.goto_target(
        head=create_head_pose(y=10, mm=True),
        antennas=np.deg2rad([-45, -45]),
        duration=2.0,
        method="cartoon",  # can be "linear", "minjerk", "ease" or "cartoon"
    )
```

If you want to test the different interpolation methods, you can run the [goto_interpolation_playground.py](../examples/goto_interpolation_playground.py) example, which will illustrate the differences between the interpolation methods when running the `goto_target` method.

You can also use the `set_target` method to set the target position of the head and antennas. It will use the same head pose and antennas arguments. This method is useful if you want the robot to move immediately to a specific position without any interpolation. It can be useful to control the movement at a high frequency. For example, to make the head follows a sinusoidal trajectory:

```python
import time
import numpy as np

from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose

with ReachyMini() as reachy:
    # Set the initial target position
    reachy.set_target(head=create_head_pose(y=0, mm=True))

    start = time.time()
    while True:
        # Calculate the elapsed time
        t = time.time() - start

        if t > 10:
            break  # Stop after 10 seconds

        # Calculate the new target position
        y = 10 * np.sin(2 * np.pi * 0.5 * t)  # Sinusoidal trajectory
        # Set the new target position
        reachy.set_target(head=create_head_pose(y=y, mm=True))

        time.sleep(0.01)
```

⚠️ <b>BEWARE: Mini's head range of motion is limited</b> ⚠️


When it comes to the reachy mini's head, it is important to note that it has several physical limitations of its head and body position and orientation 
capabilities. Such as:
1. The robots motors have a limited range of motion
2. The robot's head can collide with the body
3. The robot's body rotation (body yaw) is limited to [-180, 180] degrees.

In addition to the physical limitations, the robot has several software safety features that limit the head's position and orientation. These safety features are designed to prevent the robot from moving to unsafe positions or orientations that could cause damage to the robot or its surroundings.

4. The head's pitch and roll angles are limited to [-40, 40] degrees.
5. The head's yaw angle is limited to [-180, 180] degrees
6. The difference between the body yaw and head yaw angles is limited to [-65, 65] degrees.

If any of these limits are exceeded, the robot will move to the closest valid position within the limits.

For example, if you try to move the head to an orientation that exceeds the limits, such as:

```python
reachy.goto_target(head=create_head_pose(roll=-50, degrees=True))
```

The robot will not be able to follow due to the safety limit 4.

Or another example, if you try to move the body yaw to an angle that exceeds the limits, such as:

```python
reachy.goto_target(
    head=create_head_pose(yaw=10, degrees=True), body_yaw=np.deg2rad(100)
)
```

The robot will not be able to follow the command exactly due to the safety limit 6.

Reachy mini will not throw an error if you exceed these limits, but it will move to the closest valid position within the limits. You can check the current position of the head using the `get_current_head_pose` method of the `ReachyMini` instance. For example:

```python
from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose
import time 
reachy = ReachyMini()
# construct a head pose with roll -20 degrees
pose = create_head_pose(roll=-20, degrees=True)
reachy.goto_target(head=pose)
time.sleep(1)  # wait for the movement to complete
# get the current head pose
head_pose = reachy.get_current_head_pose()
print("current head pose", head_pose)
print("target head pose", pose)
```

### Look at

To make the robot look at a specific point, we provide look_at methods. 

The `look_at_image` method allows the robot to look at a point in the image coordinates. The image coordinates are defined as a 2D point in the camera's image plane, where (0, 0) is the top-left corner of the image and (width, height) is the bottom-right corner. Similarly to the `goto_target` method, you can specify the duration of the movement.

You can see the example in [look_at_image.py](../examples/look_at_image.py).

There is also a `look_at_world` method that allows the robot to look at a point in the world coordinates. The world coordinates are defined as a 3D point in the robot's coordinate system. TODO add a schematic of this coordinate system.

### Enable/disable motors and compliancy


You can control the robot's motors with three main methods:

1. **`enable_motors`**: Powers on the motors, making the robot ready to move and respond to commands. In this mode, the robot holds its position and you cannot move it by hand.

2. **`disable_motors`**: Powers off the motors. The robot will not respond to commands, but you can move it freely by hand (no resistance).

3. **`make_motors_compliant`**: Makes the motors compliant (requires motors to be enabled first). In this mode, the motors are powered on, but they do not resist external forces, so you can move the robot by hand, and it will feel "soft". This is useful for safe manual manipulation or teaching by demonstration. For an example, see the [gravity compensation example](../examples/reachy_compliant_demo.py): in this demo, you can move the robot's head and antennas freely, but the robot will use its motors to compensate for gravity and maintain its position/orientation in space.

### Recording moves
Let's assume that you have a script that uses the `set_target` fonction to move the robot and you want to record the commands to be able to replay them.
For example, this could be a teleoperation script, and you could be recording cute emotions (OK this is how we did it).
You can create another ReachyMini client (or use the same) and simply use `start_recording()` and `stop_recording()`.
Below is the snippet of code that show how to start a recording, stop the recording and unpack the data:
```python
    try:
        # The daemon will start recording the set_target() calls
        mini.start_recording()
        print("\nRecording started. Press Ctrl+C here to stop recording.")

        while True:
            # Keep the script alive to listen for Ctrl+C
            time.sleep(0.01) 
    except KeyboardInterrupt:
        # Stop recording and retrieve the logged data
        recorded_motion = mini.stop_recording()
        
        data = {}
        for frame in recorded_motion:
            data["time"].append(frame.get("time"))
            pose_info = {
                'head': frame.get('head'),
                'antennas': frame.get('antennas'),
                'body_yaw': frame.get('body_yaw'),
            }
            data["set_target_data"].append(pose_info)
```
## Accessing the sensors

Reachy Mini comes with several sensors (camera, microphone, speaker) that are connected to your computer via USB through the robot. These devices appear just like standard USB peripherals, so you can access them using your usual tools and libraries, exactly as if they were plugged directly into your computer.

### Camera

You can access the camera using OpenCV or any other library that supports video capture. We provide an utility function to create the capture using OpenCV:

```python
import cv2

from reachy_mini.utils.camera import find_camera

cap = find_camera()
assert cap is not None and cap.isOpened(), "Camera not found"

# Capture a frame
ret, frame = cap.read()
if not ret:
    print("Failed to capture frame")

cv2.imshow("Camera", frame)
cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows()
```

### Microphone

TODO

### Speaker

TODO

## Playing moves

You can also play predefined moves simply. You can find an example on how to do this in the [Dance Player](../examples/minimal_dance_player.py) example. 

Basically, you just need to load the move from the collections you want. And run the `play_on` method on a `ReachyMini` instance.

```python
from reachy_mini.motion.dance_move import DanceMove
from reachy_mini.reachy_mini import ReachyMini

with ReachyMini() as mini:
    move = DanceMove("dizzy_spin")
    move.play_on(mini)
```

You can also list the available moves in a collection as follows:

```python
from reachy_mini.motion.collection.dance import AVAILABLE_MOVES

print("Available moves:", list(AVAILABLE_MOVES.keys()))

>>> Available moves: ['simple_nod', 'head_tilt_roll', 'side_to_side_sway', 'dizzy_spin', 'stumble_and_recover', 'headbanger_combo', 'interwoven_spirals', 'sharp_side_tilt', 'side_peekaboo', 'yeah_nod', 'uh_huh_tilt', 'neck_recoil', 'chin_lead', 'groovy_sway_and_roll', 'chicken_peck', 'side_glance_flick', 'polyrhythm_combo', 'grid_snap', 'pendulum_swing', 'jackson_square']
```

## Writing an App

We provide a simple way to wrap your behavior in an application that can be run as a standalone script. This is useful to properly manage the start/stop of the app and to add discovery/install mechanisms to allow users to easily run your app. We are also working on a dashboard to manage the apps and their installation.

To write your app, you simply need to define a class that inherits from `ReachyMiniApp` and implements the `run` method. This method will be called when the app is started, and you can use it to interact with the robot.

```python
import threading

from reachy_mini.app import ReachyMiniApp
from reachy_mini import ReachyMini


class MyApp(ReachyMiniApp):
    def run(self, reachy_mini: ReachyMini, stop_event: threading.Event):
        # Write your code here
        ...
```

The `stop_event` is a threading event that you should check periodically to know if the app should stop. You can use it to gracefully stop the app when the user wants to stop it.

We also provide a script to make all the basic boilerplate for you. You can run the following command to create a new app:

```bash
reachy-mini-make-app my_app_name
```

This will create a new directory called `my_app_name` with the following structure:

```
my_app_name/
├── pyproject.toml
├── README.md
├── my_app_name/
│   ├── __init__.py
│   └── main.py
```

You can run your app directly as a script:

```bash
python my_app_name/main.py
```

or install it as a package:

```bash
pip install -e my_app_name/
```
