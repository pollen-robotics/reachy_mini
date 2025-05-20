# Stewart little control

IK based on https://github.com/Yeok-c/Stewart_Py

## Installation

```bash
pip install -e .
```

## Usage

Look at the examples

Dance usage:

```
python groove_test.py --mode dance --dance_name head_tilt_roll --duration 20 --bpm 124

```

## Client / server

To run the mujoco server:

```bash
mujoco-server
```

To run the robot server (runs on the real robot):

```bash
robot-server
```

To use the client :

```python
from stewart_little_control import Client
client = Client(ip)
pose = np.eye(4)
client.send_pose(pose)
```
