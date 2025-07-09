# Reachy Mini API Documentation

*⚠️ All examples shown below suppose that you have already started the Reachy Mini daemon, either by running `reachy-mini-daemon` or by using the Python module `reachy_mini.daemon.cli`. ⚠️*

## ReachyMini

Reachy Mini's API is designed to be simple and intuitive. You will mostly interact with the `ReachyMini` class, which provides methods to control the robot's joints such as the head and antennas and interacti with its sensors.

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

You can also move both the head and antennas at the same time by passing both arguments to the `goto_target` method:

```python
import numpy as np

from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose

with ReachyMini() as reachy:
    # Move the head and antennas to a specific position
    reachy.goto_target(
        head=create_head_pose(y=-10, mm=True),
        antennas=np.deg2rad([45, 45]),
        duration=2.0
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
```

### Look at

To make the robot look at a specific point, we provide look_at methods. 

The `look_at_image` method allows the robot to look at a point in the image coordinates. The image coordinates are defined as a 2D point in the camera's image plane, where (0, 0) is the top-left corner of the image and (width, height) is the bottom-right corner. Similarly to the `goto_target` method, you can specify the duration of the movement.

You can see the example in [look_at_image.py](../examples/look_at_image.py).

There is also a `look_at_world` method that allows the robot to look at a point in the world coordinates. The world coordinates are defined as a 3D point in the robot's coordinate system.

### Torque ON/OFF

You can enable/disable the torque of the motors using the `set_torque` method. This is useful to turn off the motors when you want to manipulate the robot manually or when you want to save power.

```python
from reachy_mini import ReachyMini

with ReachyMini() as reachy:
    # Disable the torque of the motors
    reachy.set_torque(False)
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

## Writing an App

TODO