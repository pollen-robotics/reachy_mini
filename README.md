# Reachy Mini

## Installation

```bash
pip install -e .
```


> TODO: Outdated. Update the readme

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
real-motors-server
```

To use the client :

```python
from stewart_little_control import Client
client = Client(ip)
pose = np.eye(4)
client.send_pose(pose)
```

## Credits

To choose a different scene:

```bash
mujoco-server -s minimal
```

### Simulation model used
https://polyhaven.com/a/food_apple_01
https://polyhaven.com/a/croissant
https://polyhaven.com/a/wooden_table_02
https://polyhaven.com/a/rubber_duck_toy