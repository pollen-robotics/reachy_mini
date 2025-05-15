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


## Mujoco server client

To run the server:
```bash
mujoco-server
```

To use the client :

```python
from stewart_little_control import MujocoClient
client = MujocoClient(ip)
pose = np.eye(4)
client.send_pose(pose)
```
