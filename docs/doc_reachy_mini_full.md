# Reachy Mini

> ⚠️ Reachy Mini is still in beta. Expect bugs, some of them we won't fix right away if they are not a priority.

[Ask questions on Hugging Face Chat](https://huggingface.co/chat/?attachments=https%3A%2F%2Fraw.githubusercontent.com%2Fpollen-robotics%2Freachy_mini%2Frefs%2Fheads%2F384-ask-questions-about-doc-on-huggingface-chat%2Fdocs%2Fdoc_reachy_mini_full.md&prompt=Read%20this%20documentation%20about%20Reachy%20Mini%20so%20I%20can%20ask%20question%20about%20it)

[Reachy Mini](https://www.pollen-robotics.com/reachy-mini/) is an expressive, open-source robot designed for human-robot interaction, creative coding, and AI experimentation. We made it to be affordable, easy to use, hackable and cute, so that you can focus on building cool AI applications!

[![Reachy Mini Hello](/docs/assets/reachy_mini_hello.gif)](https://www.pollen-robotics.com/reachy-mini/)

### Versions Lite & Wireless

Reachy Mini's hardware comes in two flavors:
- **Reachy Mini lite**: where the robot is directly connected to your computer via USB. And the code that controls the robot (the daemon) runs on your computer.
- **Reachy Mini wireless**: where an Raspberry Pi is embedded in the robot, and the code that controls the robot (the daemon) runs on the Raspberry Pi. You can connect to it via Wi-Fi from your computer (see [Wireless Setup](./docs/wireless-version.md)).

There is also a simulated version of Reachy Mini in [MuJoCo](https://mujoco.org) that you can use to prototype your applications before deploying them on the real robot. It behaves like the lite version where the daemon runs on your computer.

## Assembly guide

Follow our step-by-step [Assembly Guide](https://www.pollen-robotics.com/wp-content/uploads/2025/10/Reachy_Mini_Assembly_BETA_v2_LOW-compresse.pdf).
Most builders finish in about 3 hours, our current speed record is 43 minutes. The guide walks you through every step with clear visuals so you can assemble Reachy Mini confidently from start to finish. Enjoy the build!

## Software overview

This repository provides everything you need to control Reachy Mini, both in simulation and on the real robot. It consists of two main parts:

- **The 😈 Daemon 😈**: A background service that manages communication with the robot's motors and sensors, or with the simulation environment. It should be running before you can control the robot. It can run either for the simulation (MuJoCo) or for the real robot. 
- **🐍 SDK & 🕸️ API** to control the robot's main features (head, antennas, camera, speakers, microphone, etc.) and connect with your AI experimentation. Depending on your preferences and needs, there is a [Python SDK](#using-the-python-sdk) and a [HTTP REST API](#using-the-rest-api).

Using the [Python SDK](#using-the-python-sdk), making your robot move only require a few lines of code, as illustrated in the example below:

```python
from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose

with ReachyMini() as reachy_mini:
    # Move the head up (10mm on z-axis) and roll it 15 degrees
    pose = create_head_pose(z=10, roll=15, degrees=True, mm=True)
    reachy_mini.goto_target(head=pose, duration=2.0)

    # Reset to default pose
    pose = create_head_pose() 
    reachy_mini.goto_target(head=pose, duration=2.0)
```

and using the [REST API](#using-the-rest-api), reading the current state of the robot:

```bash
curl 'http://localhost:8000/api/state/full'
```

Those two examples above assume that the daemon is already running (either in simulation or on the real robot) locally.

## Installation of the daemon and Python SDK

As mentioned above, before being able to use the robot, you need to run the daemon that will handle the communication with the motors.

We support and test on Linux and macOS. It's also working on Windows, but it is less tested at the moment. Do not hesitate to open an issue if you encounter any problem. 

The daemon is built in Python, so you need to have Python installed on your computer (versions from 3.10 to 3.13 are supported). We recommend using a virtual environment to avoid dependency conflicts with your other Python projects.

You can install Reachy Mini from the source code or from PyPI.

First, make sure `git-lfs` is installed on your system:

- On Linux: `sudo apt install git-lfs`
- On macOS: `brew install git-lfs`
- On Windows: [Follow the instructions here](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage?platform=windows)

From PyPI, you can install the package with:

```bash
pip install reachy-mini
```

From the source code, you can install the package with:

```bash
git clone https://github.com/pollen-robotics/reachy_mini
pip install -e ./reachy_mini
```

*Note that uv users can directly run the daemon with:*
```bash
uv run reachy-mini-daemon
```

The same package provides both the daemon and the Python SDK.


## Run the reachy mini daemon

Before being able to use the robot, you need to run the daemon that will handle the communication with the motors. This daemon can run either in simulation (MuJoCo) or on the real robot.

```bash
reachy-mini-daemon
```

or run it via the Python module:

```bash
python -m reachy_mini.daemon.app.main
```

Additional argument for both simulation and real robot:

```bash
--localhost-only: (default behavior). The server will only accept connections from localhost.
```

or

```bash
--no-localhost-only: If set, the server will accept connections from any connection on the local network.
```

### In simulation ([MuJoCo](https://mujoco.org))

You first have to install the optional dependency `mujoco`.

```bash
pip install reachy-mini[mujoco]
```

Then run the daemon with the `--sim` argument.

```bash
reachy-mini-daemon --sim
```

Additional arguments:

```bash
--scene <empty|minimal> : (Default empty). Choose between a basic empty scene, or a scene with a table and some objects.
```

<img src="https://www.pollen-robotics.com/wp-content/uploads/2025/06/Reachy_mini_simulation.gif" width="250" alt="Reachy Mini in MuJoCo">


*Note: On OSX in order to run mujoco, you need to use mjpython (see [here](https://mujoco.readthedocs.io/en/stable/python.html#passive-viewer)). So, you should run the daemon with:*

```bash
 mjpython -m reachy_mini.daemon.app.main --sim
 ```

### For the lite version (connected via USB)

It should automatically detect the serial port of the robot. If it does not, you can specify it manually with the `-p` option:

```bash
reachy-mini-daemon -p <serial_port>
```

### Usage

For more information about the daemon and its options, you can run:

```bash
reachy-mini-daemon --help
```

### Dashboard

You can access a simple dashboard to monitor the robot's status at [http://localhost:8000/](http://localhost:8000/) when the daemon is running. This lets you turn your robot on and off, run some basic movements, and browse spaces for Reachy Mini!

![Reachy Mini Dashboard](docs/assets/dashboard.png)

## Run the demo & awesome apps

Conversational demo for the Reachy Mini robot combining LLM realtime APIs, vision pipelines, and choreographed motion libraries: [reachy_mini_conversation_demo](https://github.com/pollen-robotics/reachy_mini_conversation_demo).

You can find more awesome apps and demos for Reachy Mini on [Hugging Face spaces](https://huggingface.co/spaces?q=reachy_mini)!

## Using the Python SDK

The API is designed to be simple and intuitive. You can control the robot's features such as the head, antennas, camera, speakers, and microphone. For instance, to move the head of the robot, you can use the `goto_target` method as shown in the example below:

```python
from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose

with ReachyMini() as reachy_mini:
    # Move the head up (10mm on z-axis) and roll it 15 degrees
    pose = create_head_pose(z=10, roll=15, degrees=True, mm=True)
    reachy_mini.goto_target(head=pose, duration=2.0)

    # Reset to default pose
    pose = create_head_pose() 
    reachy_mini.goto_target(head=pose, duration=2.0)
```

For a full description of the SDK, please refer to the [Python SDK documentation](./docs/python-sdk.md).

## Using the REST API

The daemon also provides a REST API via [fastapi](https://fastapi.tiangolo.com/) that you can use to control the robot and get its state. The API is accessible via HTTP and WebSocket.

By default, the API server runs on `http://localhost:8000`. The API is documented using OpenAPI, and you can access the documentation at `http://localhost:8000/docs` when the daemon is running.

More information about the API can be found in the [HTTP API documentation](./docs/rest-api.md).

## Open source & contribution

This project is actively developed and maintained by the [Pollen Robotics team](https://www.pollen-robotics.com) and the [Hugging Face team](https://huggingface.co/). 

We welcome contributions from the community! If you want to report a bug or request a feature, please open an issue on GitHub. If you want to contribute code, please fork the repository and submit a pull request.

### 3D models

TODO

### Contributing

Development tools are available in the optional dependencies.

```bash
pip install -e .[dev]
pre-commit install
```

Your files will be checked before any commit. Checks may also be manually run with

```bash
pre-commit run --all-files
```

Checks are performed by Ruff. You may want to [configure your IDE to support it](https://docs.astral.sh/ruff/editors/setup/).

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

The robot design files are licensed under the [TODO](TODO) license.

### Simulation model used

- https://polyhaven.com/a/food_apple_01
- https://polyhaven.com/a/croissant
- https://polyhaven.com/a/wooden_table_02
- https://polyhaven.com/a/rubber_duck_toy
# Table of Contents

* [reachy\_mini](#reachy_mini)
* [reachy\_mini.reachy\_mini](#reachy_mini.reachy_mini)
  * [ReachyMini](#reachy_mini.reachy_mini.ReachyMini)
    * [\_\_init\_\_](#reachy_mini.reachy_mini.ReachyMini.__init__)
    * [\_\_del\_\_](#reachy_mini.reachy_mini.ReachyMini.__del__)
    * [\_\_enter\_\_](#reachy_mini.reachy_mini.ReachyMini.__enter__)
    * [\_\_exit\_\_](#reachy_mini.reachy_mini.ReachyMini.__exit__)
    * [media](#reachy_mini.reachy_mini.ReachyMini.media)
    * [set\_target](#reachy_mini.reachy_mini.ReachyMini.set_target)
    * [goto\_target](#reachy_mini.reachy_mini.ReachyMini.goto_target)
    * [wake\_up](#reachy_mini.reachy_mini.ReachyMini.wake_up)
    * [goto\_sleep](#reachy_mini.reachy_mini.ReachyMini.goto_sleep)
    * [look\_at\_image](#reachy_mini.reachy_mini.ReachyMini.look_at_image)
    * [look\_at\_world](#reachy_mini.reachy_mini.ReachyMini.look_at_world)
    * [get\_current\_joint\_positions](#reachy_mini.reachy_mini.ReachyMini.get_current_joint_positions)
    * [get\_present\_antenna\_joint\_positions](#reachy_mini.reachy_mini.ReachyMini.get_present_antenna_joint_positions)
    * [get\_current\_head\_pose](#reachy_mini.reachy_mini.ReachyMini.get_current_head_pose)
    * [set\_target\_head\_pose](#reachy_mini.reachy_mini.ReachyMini.set_target_head_pose)
    * [set\_target\_antenna\_joint\_positions](#reachy_mini.reachy_mini.ReachyMini.set_target_antenna_joint_positions)
    * [set\_target\_body\_yaw](#reachy_mini.reachy_mini.ReachyMini.set_target_body_yaw)
    * [start\_recording](#reachy_mini.reachy_mini.ReachyMini.start_recording)
    * [stop\_recording](#reachy_mini.reachy_mini.ReachyMini.stop_recording)
    * [enable\_motors](#reachy_mini.reachy_mini.ReachyMini.enable_motors)
    * [disable\_motors](#reachy_mini.reachy_mini.ReachyMini.disable_motors)
    * [enable\_gravity\_compensation](#reachy_mini.reachy_mini.ReachyMini.enable_gravity_compensation)
    * [disable\_gravity\_compensation](#reachy_mini.reachy_mini.ReachyMini.disable_gravity_compensation)
    * [set\_automatic\_body\_yaw](#reachy_mini.reachy_mini.ReachyMini.set_automatic_body_yaw)
    * [async\_play\_move](#reachy_mini.reachy_mini.ReachyMini.async_play_move)
* [reachy\_mini.kinematics.analytical\_kinematics](#reachy_mini.kinematics.analytical_kinematics)
  * [AnalyticalKinematics](#reachy_mini.kinematics.analytical_kinematics.AnalyticalKinematics)
    * [\_\_init\_\_](#reachy_mini.kinematics.analytical_kinematics.AnalyticalKinematics.__init__)
    * [ik](#reachy_mini.kinematics.analytical_kinematics.AnalyticalKinematics.ik)
    * [fk](#reachy_mini.kinematics.analytical_kinematics.AnalyticalKinematics.fk)
* [reachy\_mini.kinematics.placo\_kinematics](#reachy_mini.kinematics.placo_kinematics)
  * [PlacoKinematics](#reachy_mini.kinematics.placo_kinematics.PlacoKinematics)
    * [\_\_init\_\_](#reachy_mini.kinematics.placo_kinematics.PlacoKinematics.__init__)
    * [ik](#reachy_mini.kinematics.placo_kinematics.PlacoKinematics.ik)
    * [fk](#reachy_mini.kinematics.placo_kinematics.PlacoKinematics.fk)
    * [config\_collision\_model](#reachy_mini.kinematics.placo_kinematics.PlacoKinematics.config_collision_model)
    * [compute\_collision](#reachy_mini.kinematics.placo_kinematics.PlacoKinematics.compute_collision)
    * [compute\_jacobian](#reachy_mini.kinematics.placo_kinematics.PlacoKinematics.compute_jacobian)
    * [compute\_gravity\_torque](#reachy_mini.kinematics.placo_kinematics.PlacoKinematics.compute_gravity_torque)
    * [set\_automatic\_body\_yaw](#reachy_mini.kinematics.placo_kinematics.PlacoKinematics.set_automatic_body_yaw)
    * [get\_joint](#reachy_mini.kinematics.placo_kinematics.PlacoKinematics.get_joint)
* [reachy\_mini.kinematics](#reachy_mini.kinematics)
* [reachy\_mini.kinematics.nn\_kinematics](#reachy_mini.kinematics.nn_kinematics)
  * [NNKinematics](#reachy_mini.kinematics.nn_kinematics.NNKinematics)
    * [\_\_init\_\_](#reachy_mini.kinematics.nn_kinematics.NNKinematics.__init__)
    * [ik](#reachy_mini.kinematics.nn_kinematics.NNKinematics.ik)
    * [fk](#reachy_mini.kinematics.nn_kinematics.NNKinematics.fk)
  * [OnnxInfer](#reachy_mini.kinematics.nn_kinematics.OnnxInfer)
    * [\_\_init\_\_](#reachy_mini.kinematics.nn_kinematics.OnnxInfer.__init__)
    * [infer](#reachy_mini.kinematics.nn_kinematics.OnnxInfer.infer)
* [reachy\_mini.utils.constants](#reachy_mini.utils.constants)
* [reachy\_mini.utils.wireless\_version.utils](#reachy_mini.utils.wireless_version.utils)
  * [call\_logger\_wrapper](#reachy_mini.utils.wireless_version.utils.call_logger_wrapper)
* [reachy\_mini.utils.wireless\_version.update](#reachy_mini.utils.wireless_version.update)
  * [update\_reachy\_mini](#reachy_mini.utils.wireless_version.update.update_reachy_mini)
* [reachy\_mini.utils.wireless\_version](#reachy_mini.utils.wireless_version)
* [reachy\_mini.utils.wireless\_version.update\_available](#reachy_mini.utils.wireless_version.update_available)
  * [is\_update\_available](#reachy_mini.utils.wireless_version.update_available.is_update_available)
  * [get\_pypi\_version](#reachy_mini.utils.wireless_version.update_available.get_pypi_version)
  * [get\_local\_version](#reachy_mini.utils.wireless_version.update_available.get_local_version)
* [reachy\_mini.utils.interpolation](#reachy_mini.utils.interpolation)
  * [minimum\_jerk](#reachy_mini.utils.interpolation.minimum_jerk)
  * [linear\_pose\_interpolation](#reachy_mini.utils.interpolation.linear_pose_interpolation)
  * [InterpolationTechnique](#reachy_mini.utils.interpolation.InterpolationTechnique)
  * [time\_trajectory](#reachy_mini.utils.interpolation.time_trajectory)
  * [delta\_angle\_between\_mat\_rot](#reachy_mini.utils.interpolation.delta_angle_between_mat_rot)
  * [distance\_between\_poses](#reachy_mini.utils.interpolation.distance_between_poses)
  * [compose\_world\_offset](#reachy_mini.utils.interpolation.compose_world_offset)
* [reachy\_mini.utils.rerun](#reachy_mini.utils.rerun)
  * [Rerun](#reachy_mini.utils.rerun.Rerun)
    * [\_\_init\_\_](#reachy_mini.utils.rerun.Rerun.__init__)
    * [set\_absolute\_path\_to\_urdf](#reachy_mini.utils.rerun.Rerun.set_absolute_path_to_urdf)
    * [start](#reachy_mini.utils.rerun.Rerun.start)
    * [stop](#reachy_mini.utils.rerun.Rerun.stop)
    * [log\_camera](#reachy_mini.utils.rerun.Rerun.log_camera)
    * [log\_movements](#reachy_mini.utils.rerun.Rerun.log_movements)
* [reachy\_mini.utils](#reachy_mini.utils)
  * [create\_head\_pose](#reachy_mini.utils.create_head_pose)
* [reachy\_mini.utils.parse\_urdf\_for\_kinematics](#reachy_mini.utils.parse_urdf_for_kinematics)
  * [get\_data](#reachy_mini.utils.parse_urdf_for_kinematics.get_data)
  * [main](#reachy_mini.utils.parse_urdf_for_kinematics.main)
* [reachy\_mini.apps.utils](#reachy_mini.apps.utils)
  * [running\_command](#reachy_mini.apps.utils.running_command)
* [reachy\_mini.apps.sources.hf\_space](#reachy_mini.apps.sources.hf_space)
  * [list\_available\_apps](#reachy_mini.apps.sources.hf_space.list_available_apps)
* [reachy\_mini.apps.sources](#reachy_mini.apps.sources)
* [reachy\_mini.apps.sources.local\_common\_venv](#reachy_mini.apps.sources.local_common_venv)
  * [list\_available\_apps](#reachy_mini.apps.sources.local_common_venv.list_available_apps)
  * [install\_package](#reachy_mini.apps.sources.local_common_venv.install_package)
  * [uninstall\_package](#reachy_mini.apps.sources.local_common_venv.uninstall_package)
* [reachy\_mini.apps.manager](#reachy_mini.apps.manager)
  * [AppState](#reachy_mini.apps.manager.AppState)
  * [AppStatus](#reachy_mini.apps.manager.AppStatus)
  * [RunningApp](#reachy_mini.apps.manager.RunningApp)
  * [AppManager](#reachy_mini.apps.manager.AppManager)
    * [\_\_init\_\_](#reachy_mini.apps.manager.AppManager.__init__)
    * [close](#reachy_mini.apps.manager.AppManager.close)
    * [is\_app\_running](#reachy_mini.apps.manager.AppManager.is_app_running)
    * [start\_app](#reachy_mini.apps.manager.AppManager.start_app)
    * [stop\_current\_app](#reachy_mini.apps.manager.AppManager.stop_current_app)
    * [restart\_current\_app](#reachy_mini.apps.manager.AppManager.restart_current_app)
    * [current\_app\_status](#reachy_mini.apps.manager.AppManager.current_app_status)
    * [list\_all\_available\_apps](#reachy_mini.apps.manager.AppManager.list_all_available_apps)
    * [list\_available\_apps](#reachy_mini.apps.manager.AppManager.list_available_apps)
    * [install\_new\_app](#reachy_mini.apps.manager.AppManager.install_new_app)
    * [remove\_app](#reachy_mini.apps.manager.AppManager.remove_app)
* [reachy\_mini.apps.app](#reachy_mini.apps.app)
  * [ReachyMiniApp](#reachy_mini.apps.app.ReachyMiniApp)
    * [\_\_init\_\_](#reachy_mini.apps.app.ReachyMiniApp.__init__)
    * [wrapped\_run](#reachy_mini.apps.app.ReachyMiniApp.wrapped_run)
    * [run](#reachy_mini.apps.app.ReachyMiniApp.run)
    * [stop](#reachy_mini.apps.app.ReachyMiniApp.stop)
  * [make\_app\_project](#reachy_mini.apps.app.make_app_project)
  * [main](#reachy_mini.apps.app.main)
* [reachy\_mini.apps](#reachy_mini.apps)
  * [SourceKind](#reachy_mini.apps.SourceKind)
  * [AppInfo](#reachy_mini.apps.AppInfo)
* [reachy\_mini.media.camera\_gstreamer](#reachy_mini.media.camera_gstreamer)
  * [GStreamerCamera](#reachy_mini.media.camera_gstreamer.GStreamerCamera)
    * [\_\_init\_\_](#reachy_mini.media.camera_gstreamer.GStreamerCamera.__init__)
    * [open](#reachy_mini.media.camera_gstreamer.GStreamerCamera.open)
    * [read](#reachy_mini.media.camera_gstreamer.GStreamerCamera.read)
    * [close](#reachy_mini.media.camera_gstreamer.GStreamerCamera.close)
    * [get\_arducam\_video\_device](#reachy_mini.media.camera_gstreamer.GStreamerCamera.get_arducam_video_device)
* [reachy\_mini.media.webrtc\_client\_gstreamer](#reachy_mini.media.webrtc_client_gstreamer)
  * [GstWebRTCClient](#reachy_mini.media.webrtc_client_gstreamer.GstWebRTCClient)
    * [\_\_init\_\_](#reachy_mini.media.webrtc_client_gstreamer.GstWebRTCClient.__init__)
    * [\_\_del\_\_](#reachy_mini.media.webrtc_client_gstreamer.GstWebRTCClient.__del__)
    * [open](#reachy_mini.media.webrtc_client_gstreamer.GstWebRTCClient.open)
    * [get\_audio\_sample](#reachy_mini.media.webrtc_client_gstreamer.GstWebRTCClient.get_audio_sample)
    * [read](#reachy_mini.media.webrtc_client_gstreamer.GstWebRTCClient.read)
    * [close](#reachy_mini.media.webrtc_client_gstreamer.GstWebRTCClient.close)
    * [start\_recording](#reachy_mini.media.webrtc_client_gstreamer.GstWebRTCClient.start_recording)
    * [stop\_recording](#reachy_mini.media.webrtc_client_gstreamer.GstWebRTCClient.stop_recording)
    * [start\_playing](#reachy_mini.media.webrtc_client_gstreamer.GstWebRTCClient.start_playing)
    * [stop\_playing](#reachy_mini.media.webrtc_client_gstreamer.GstWebRTCClient.stop_playing)
    * [push\_audio\_sample](#reachy_mini.media.webrtc_client_gstreamer.GstWebRTCClient.push_audio_sample)
    * [play\_sound](#reachy_mini.media.webrtc_client_gstreamer.GstWebRTCClient.play_sound)
* [reachy\_mini.media.media\_manager](#reachy_mini.media.media_manager)
  * [MediaBackend](#reachy_mini.media.media_manager.MediaBackend)
  * [MediaManager](#reachy_mini.media.media_manager.MediaManager)
    * [\_\_init\_\_](#reachy_mini.media.media_manager.MediaManager.__init__)
    * [\_\_del\_\_](#reachy_mini.media.media_manager.MediaManager.__del__)
    * [get\_frame](#reachy_mini.media.media_manager.MediaManager.get_frame)
    * [play\_sound](#reachy_mini.media.media_manager.MediaManager.play_sound)
    * [start\_recording](#reachy_mini.media.media_manager.MediaManager.start_recording)
    * [get\_audio\_sample](#reachy_mini.media.media_manager.MediaManager.get_audio_sample)
    * [get\_audio\_samplerate](#reachy_mini.media.media_manager.MediaManager.get_audio_samplerate)
    * [stop\_recording](#reachy_mini.media.media_manager.MediaManager.stop_recording)
    * [start\_playing](#reachy_mini.media.media_manager.MediaManager.start_playing)
    * [push\_audio\_sample](#reachy_mini.media.media_manager.MediaManager.push_audio_sample)
    * [stop\_playing](#reachy_mini.media.media_manager.MediaManager.stop_playing)
* [reachy\_mini.media.camera\_utils](#reachy_mini.media.camera_utils)
  * [find\_camera](#reachy_mini.media.camera_utils.find_camera)
* [reachy\_mini.media.camera\_constants](#reachy_mini.media.camera_constants)
  * [CameraResolution](#reachy_mini.media.camera_constants.CameraResolution)
  * [RPICameraResolution](#reachy_mini.media.camera_constants.RPICameraResolution)
* [reachy\_mini.media.camera\_opencv](#reachy_mini.media.camera_opencv)
  * [OpenCVCamera](#reachy_mini.media.camera_opencv.OpenCVCamera)
    * [\_\_init\_\_](#reachy_mini.media.camera_opencv.OpenCVCamera.__init__)
    * [open](#reachy_mini.media.camera_opencv.OpenCVCamera.open)
    * [read](#reachy_mini.media.camera_opencv.OpenCVCamera.read)
    * [close](#reachy_mini.media.camera_opencv.OpenCVCamera.close)
* [reachy\_mini.media.webrtc\_daemon](#reachy_mini.media.webrtc_daemon)
  * [GstWebRTC](#reachy_mini.media.webrtc_daemon.GstWebRTC)
    * [\_\_init\_\_](#reachy_mini.media.webrtc_daemon.GstWebRTC.__init__)
    * [\_\_del\_\_](#reachy_mini.media.webrtc_daemon.GstWebRTC.__del__)
    * [resolution](#reachy_mini.media.webrtc_daemon.GstWebRTC.resolution)
    * [framerate](#reachy_mini.media.webrtc_daemon.GstWebRTC.framerate)
    * [start](#reachy_mini.media.webrtc_daemon.GstWebRTC.start)
    * [pause](#reachy_mini.media.webrtc_daemon.GstWebRTC.pause)
    * [stop](#reachy_mini.media.webrtc_daemon.GstWebRTC.stop)
* [reachy\_mini.media.audio\_base](#reachy_mini.media.audio_base)
  * [AudioBase](#reachy_mini.media.audio_base.AudioBase)
    * [SAMPLE\_RATE](#reachy_mini.media.audio_base.AudioBase.SAMPLE_RATE)
    * [\_\_init\_\_](#reachy_mini.media.audio_base.AudioBase.__init__)
    * [\_\_del\_\_](#reachy_mini.media.audio_base.AudioBase.__del__)
    * [start\_recording](#reachy_mini.media.audio_base.AudioBase.start_recording)
    * [get\_audio\_sample](#reachy_mini.media.audio_base.AudioBase.get_audio_sample)
    * [stop\_recording](#reachy_mini.media.audio_base.AudioBase.stop_recording)
    * [start\_playing](#reachy_mini.media.audio_base.AudioBase.start_playing)
    * [push\_audio\_sample](#reachy_mini.media.audio_base.AudioBase.push_audio_sample)
    * [stop\_playing](#reachy_mini.media.audio_base.AudioBase.stop_playing)
    * [play\_sound](#reachy_mini.media.audio_base.AudioBase.play_sound)
    * [get\_DoA](#reachy_mini.media.audio_base.AudioBase.get_DoA)
* [reachy\_mini.media.audio\_gstreamer](#reachy_mini.media.audio_gstreamer)
  * [GStreamerAudio](#reachy_mini.media.audio_gstreamer.GStreamerAudio)
    * [\_\_init\_\_](#reachy_mini.media.audio_gstreamer.GStreamerAudio.__init__)
    * [\_\_del\_\_](#reachy_mini.media.audio_gstreamer.GStreamerAudio.__del__)
    * [start\_recording](#reachy_mini.media.audio_gstreamer.GStreamerAudio.start_recording)
    * [get\_audio\_sample](#reachy_mini.media.audio_gstreamer.GStreamerAudio.get_audio_sample)
    * [stop\_recording](#reachy_mini.media.audio_gstreamer.GStreamerAudio.stop_recording)
    * [start\_playing](#reachy_mini.media.audio_gstreamer.GStreamerAudio.start_playing)
    * [stop\_playing](#reachy_mini.media.audio_gstreamer.GStreamerAudio.stop_playing)
    * [push\_audio\_sample](#reachy_mini.media.audio_gstreamer.GStreamerAudio.push_audio_sample)
    * [play\_sound](#reachy_mini.media.audio_gstreamer.GStreamerAudio.play_sound)
* [reachy\_mini.media.audio\_sounddevice](#reachy_mini.media.audio_sounddevice)
  * [SoundDeviceAudio](#reachy_mini.media.audio_sounddevice.SoundDeviceAudio)
    * [\_\_init\_\_](#reachy_mini.media.audio_sounddevice.SoundDeviceAudio.__init__)
    * [start\_recording](#reachy_mini.media.audio_sounddevice.SoundDeviceAudio.start_recording)
    * [get\_audio\_sample](#reachy_mini.media.audio_sounddevice.SoundDeviceAudio.get_audio_sample)
    * [stop\_recording](#reachy_mini.media.audio_sounddevice.SoundDeviceAudio.stop_recording)
    * [push\_audio\_sample](#reachy_mini.media.audio_sounddevice.SoundDeviceAudio.push_audio_sample)
    * [start\_playing](#reachy_mini.media.audio_sounddevice.SoundDeviceAudio.start_playing)
    * [stop\_playing](#reachy_mini.media.audio_sounddevice.SoundDeviceAudio.stop_playing)
    * [play\_sound](#reachy_mini.media.audio_sounddevice.SoundDeviceAudio.play_sound)
    * [get\_output\_device\_id](#reachy_mini.media.audio_sounddevice.SoundDeviceAudio.get_output_device_id)
    * [get\_input\_device\_id](#reachy_mini.media.audio_sounddevice.SoundDeviceAudio.get_input_device_id)
* [reachy\_mini.media.camera\_base](#reachy_mini.media.camera_base)
  * [CameraBase](#reachy_mini.media.camera_base.CameraBase)
    * [\_\_init\_\_](#reachy_mini.media.camera_base.CameraBase.__init__)
    * [resolution](#reachy_mini.media.camera_base.CameraBase.resolution)
    * [framerate](#reachy_mini.media.camera_base.CameraBase.framerate)
    * [open](#reachy_mini.media.camera_base.CameraBase.open)
    * [read](#reachy_mini.media.camera_base.CameraBase.read)
    * [close](#reachy_mini.media.camera_base.CameraBase.close)
* [reachy\_mini.media.audio\_utils](#reachy_mini.media.audio_utils)
  * [get\_respeaker\_card\_number](#reachy_mini.media.audio_utils.get_respeaker_card_number)
* [reachy\_mini.media](#reachy_mini.media)
* [reachy\_mini.io.abstract](#reachy_mini.io.abstract)
  * [AbstractServer](#reachy_mini.io.abstract.AbstractServer)
    * [start](#reachy_mini.io.abstract.AbstractServer.start)
    * [stop](#reachy_mini.io.abstract.AbstractServer.stop)
    * [command\_received\_event](#reachy_mini.io.abstract.AbstractServer.command_received_event)
  * [AbstractClient](#reachy_mini.io.abstract.AbstractClient)
    * [wait\_for\_connection](#reachy_mini.io.abstract.AbstractClient.wait_for_connection)
    * [is\_connected](#reachy_mini.io.abstract.AbstractClient.is_connected)
    * [disconnect](#reachy_mini.io.abstract.AbstractClient.disconnect)
    * [send\_command](#reachy_mini.io.abstract.AbstractClient.send_command)
    * [get\_current\_joints](#reachy_mini.io.abstract.AbstractClient.get_current_joints)
    * [send\_task\_request](#reachy_mini.io.abstract.AbstractClient.send_task_request)
    * [wait\_for\_task\_completion](#reachy_mini.io.abstract.AbstractClient.wait_for_task_completion)
* [reachy\_mini.io.zenoh\_client](#reachy_mini.io.zenoh_client)
  * [ZenohClient](#reachy_mini.io.zenoh_client.ZenohClient)
    * [\_\_init\_\_](#reachy_mini.io.zenoh_client.ZenohClient.__init__)
    * [wait\_for\_connection](#reachy_mini.io.zenoh_client.ZenohClient.wait_for_connection)
    * [check\_alive](#reachy_mini.io.zenoh_client.ZenohClient.check_alive)
    * [is\_connected](#reachy_mini.io.zenoh_client.ZenohClient.is_connected)
    * [disconnect](#reachy_mini.io.zenoh_client.ZenohClient.disconnect)
    * [send\_command](#reachy_mini.io.zenoh_client.ZenohClient.send_command)
    * [get\_current\_joints](#reachy_mini.io.zenoh_client.ZenohClient.get_current_joints)
    * [wait\_for\_recorded\_data](#reachy_mini.io.zenoh_client.ZenohClient.wait_for_recorded_data)
    * [get\_recorded\_data](#reachy_mini.io.zenoh_client.ZenohClient.get_recorded_data)
    * [get\_status](#reachy_mini.io.zenoh_client.ZenohClient.get_status)
    * [get\_current\_head\_pose](#reachy_mini.io.zenoh_client.ZenohClient.get_current_head_pose)
    * [send\_task\_request](#reachy_mini.io.zenoh_client.ZenohClient.send_task_request)
    * [wait\_for\_task\_completion](#reachy_mini.io.zenoh_client.ZenohClient.wait_for_task_completion)
  * [TaskState](#reachy_mini.io.zenoh_client.TaskState)
* [reachy\_mini.io](#reachy_mini.io)
* [reachy\_mini.io.zenoh\_server](#reachy_mini.io.zenoh_server)
  * [ZenohServer](#reachy_mini.io.zenoh_server.ZenohServer)
    * [\_\_init\_\_](#reachy_mini.io.zenoh_server.ZenohServer.__init__)
    * [start](#reachy_mini.io.zenoh_server.ZenohServer.start)
    * [stop](#reachy_mini.io.zenoh_server.ZenohServer.stop)
    * [command\_received\_event](#reachy_mini.io.zenoh_server.ZenohServer.command_received_event)
* [reachy\_mini.io.protocol](#reachy_mini.io.protocol)
  * [GotoTaskRequest](#reachy_mini.io.protocol.GotoTaskRequest)
    * [head](#reachy_mini.io.protocol.GotoTaskRequest.head)
    * [antennas](#reachy_mini.io.protocol.GotoTaskRequest.antennas)
  * [PlayMoveTaskRequest](#reachy_mini.io.protocol.PlayMoveTaskRequest)
  * [TaskRequest](#reachy_mini.io.protocol.TaskRequest)
  * [TaskProgress](#reachy_mini.io.protocol.TaskProgress)
* [reachy\_mini.daemon.utils](#reachy_mini.daemon.utils)
  * [daemon\_check](#reachy_mini.daemon.utils.daemon_check)
  * [find\_serial\_port](#reachy_mini.daemon.utils.find_serial_port)
  * [get\_ip\_address](#reachy_mini.daemon.utils.get_ip_address)
  * [convert\_enum\_to\_dict](#reachy_mini.daemon.utils.convert_enum_to_dict)
* [reachy\_mini.daemon.app.routers.move](#reachy_mini.daemon.app.routers.move)
  * [InterpolationMode](#reachy_mini.daemon.app.routers.move.InterpolationMode)
  * [GotoModelRequest](#reachy_mini.daemon.app.routers.move.GotoModelRequest)
  * [MoveUUID](#reachy_mini.daemon.app.routers.move.MoveUUID)
  * [create\_move\_task](#reachy_mini.daemon.app.routers.move.create_move_task)
  * [stop\_move\_task](#reachy_mini.daemon.app.routers.move.stop_move_task)
  * [get\_running\_moves](#reachy_mini.daemon.app.routers.move.get_running_moves)
  * [goto](#reachy_mini.daemon.app.routers.move.goto)
  * [play\_wake\_up](#reachy_mini.daemon.app.routers.move.play_wake_up)
  * [play\_goto\_sleep](#reachy_mini.daemon.app.routers.move.play_goto_sleep)
  * [list\_recorded\_move\_dataset](#reachy_mini.daemon.app.routers.move.list_recorded_move_dataset)
  * [play\_recorded\_move\_dataset](#reachy_mini.daemon.app.routers.move.play_recorded_move_dataset)
  * [stop\_move](#reachy_mini.daemon.app.routers.move.stop_move)
  * [ws\_move\_updates](#reachy_mini.daemon.app.routers.move.ws_move_updates)
  * [set\_target](#reachy_mini.daemon.app.routers.move.set_target)
  * [ws\_set\_target](#reachy_mini.daemon.app.routers.move.ws_set_target)
* [reachy\_mini.daemon.app.routers.wifi\_config](#reachy_mini.daemon.app.routers.wifi_config)
  * [WifiMode](#reachy_mini.daemon.app.routers.wifi_config.WifiMode)
  * [WifiStatus](#reachy_mini.daemon.app.routers.wifi_config.WifiStatus)
  * [get\_current\_wifi\_mode](#reachy_mini.daemon.app.routers.wifi_config.get_current_wifi_mode)
  * [get\_wifi\_status](#reachy_mini.daemon.app.routers.wifi_config.get_wifi_status)
  * [get\_last\_wifi\_error](#reachy_mini.daemon.app.routers.wifi_config.get_last_wifi_error)
  * [reset\_last\_wifi\_error](#reachy_mini.daemon.app.routers.wifi_config.reset_last_wifi_error)
  * [setup\_hotspot](#reachy_mini.daemon.app.routers.wifi_config.setup_hotspot)
  * [connect\_to\_wifi\_network](#reachy_mini.daemon.app.routers.wifi_config.connect_to_wifi_network)
  * [scan\_wifi](#reachy_mini.daemon.app.routers.wifi_config.scan_wifi)
  * [scan\_available\_wifi](#reachy_mini.daemon.app.routers.wifi_config.scan_available_wifi)
  * [get\_wifi\_connections](#reachy_mini.daemon.app.routers.wifi_config.get_wifi_connections)
  * [check\_if\_connection\_exists](#reachy_mini.daemon.app.routers.wifi_config.check_if_connection_exists)
  * [check\_if\_connection\_active](#reachy_mini.daemon.app.routers.wifi_config.check_if_connection_active)
  * [setup\_wifi\_connection](#reachy_mini.daemon.app.routers.wifi_config.setup_wifi_connection)
  * [remove\_connection](#reachy_mini.daemon.app.routers.wifi_config.remove_connection)
* [reachy\_mini.daemon.app.routers.update](#reachy_mini.daemon.app.routers.update)
  * [available](#reachy_mini.daemon.app.routers.update.available)
  * [start\_update](#reachy_mini.daemon.app.routers.update.start_update)
  * [get\_update\_info](#reachy_mini.daemon.app.routers.update.get_update_info)
  * [websocket\_logs](#reachy_mini.daemon.app.routers.update.websocket_logs)
* [reachy\_mini.daemon.app.routers.state](#reachy_mini.daemon.app.routers.state)
  * [get\_head\_pose](#reachy_mini.daemon.app.routers.state.get_head_pose)
  * [get\_body\_yaw](#reachy_mini.daemon.app.routers.state.get_body_yaw)
  * [get\_antenna\_joint\_positions](#reachy_mini.daemon.app.routers.state.get_antenna_joint_positions)
  * [get\_full\_state](#reachy_mini.daemon.app.routers.state.get_full_state)
  * [ws\_full\_state](#reachy_mini.daemon.app.routers.state.ws_full_state)
* [reachy\_mini.daemon.app.routers.apps](#reachy_mini.daemon.app.routers.apps)
  * [list\_available\_apps](#reachy_mini.daemon.app.routers.apps.list_available_apps)
  * [list\_all\_available\_apps](#reachy_mini.daemon.app.routers.apps.list_all_available_apps)
  * [install\_app](#reachy_mini.daemon.app.routers.apps.install_app)
  * [remove\_app](#reachy_mini.daemon.app.routers.apps.remove_app)
  * [job\_status](#reachy_mini.daemon.app.routers.apps.job_status)
  * [ws\_apps\_manager](#reachy_mini.daemon.app.routers.apps.ws_apps_manager)
  * [start\_app](#reachy_mini.daemon.app.routers.apps.start_app)
  * [restart\_app](#reachy_mini.daemon.app.routers.apps.restart_app)
  * [stop\_app](#reachy_mini.daemon.app.routers.apps.stop_app)
  * [current\_app\_status](#reachy_mini.daemon.app.routers.apps.current_app_status)
* [reachy\_mini.daemon.app.routers.kinematics](#reachy_mini.daemon.app.routers.kinematics)
  * [get\_kinematics\_info](#reachy_mini.daemon.app.routers.kinematics.get_kinematics_info)
  * [get\_urdf](#reachy_mini.daemon.app.routers.kinematics.get_urdf)
  * [get\_stl\_file](#reachy_mini.daemon.app.routers.kinematics.get_stl_file)
* [reachy\_mini.daemon.app.routers.daemon](#reachy_mini.daemon.app.routers.daemon)
  * [start\_daemon](#reachy_mini.daemon.app.routers.daemon.start_daemon)
  * [stop\_daemon](#reachy_mini.daemon.app.routers.daemon.stop_daemon)
  * [restart\_daemon](#reachy_mini.daemon.app.routers.daemon.restart_daemon)
  * [get\_daemon\_status](#reachy_mini.daemon.app.routers.daemon.get_daemon_status)
* [reachy\_mini.daemon.app.routers.motors](#reachy_mini.daemon.app.routers.motors)
  * [MotorStatus](#reachy_mini.daemon.app.routers.motors.MotorStatus)
  * [get\_motor\_status](#reachy_mini.daemon.app.routers.motors.get_motor_status)
  * [set\_motor\_mode](#reachy_mini.daemon.app.routers.motors.set_motor_mode)
* [reachy\_mini.daemon.app.services.bluetooth.bluetooth\_service](#reachy_mini.daemon.app.services.bluetooth.bluetooth_service)
  * [NoInputAgent](#reachy_mini.daemon.app.services.bluetooth.bluetooth_service.NoInputAgent)
    * [Release](#reachy_mini.daemon.app.services.bluetooth.bluetooth_service.NoInputAgent.Release)
    * [RequestPinCode](#reachy_mini.daemon.app.services.bluetooth.bluetooth_service.NoInputAgent.RequestPinCode)
    * [RequestPasskey](#reachy_mini.daemon.app.services.bluetooth.bluetooth_service.NoInputAgent.RequestPasskey)
    * [RequestConfirmation](#reachy_mini.daemon.app.services.bluetooth.bluetooth_service.NoInputAgent.RequestConfirmation)
    * [DisplayPinCode](#reachy_mini.daemon.app.services.bluetooth.bluetooth_service.NoInputAgent.DisplayPinCode)
    * [DisplayPasskey](#reachy_mini.daemon.app.services.bluetooth.bluetooth_service.NoInputAgent.DisplayPasskey)
    * [AuthorizeService](#reachy_mini.daemon.app.services.bluetooth.bluetooth_service.NoInputAgent.AuthorizeService)
    * [Cancel](#reachy_mini.daemon.app.services.bluetooth.bluetooth_service.NoInputAgent.Cancel)
  * [Advertisement](#reachy_mini.daemon.app.services.bluetooth.bluetooth_service.Advertisement)
    * [\_\_init\_\_](#reachy_mini.daemon.app.services.bluetooth.bluetooth_service.Advertisement.__init__)
    * [get\_properties](#reachy_mini.daemon.app.services.bluetooth.bluetooth_service.Advertisement.get_properties)
    * [get\_path](#reachy_mini.daemon.app.services.bluetooth.bluetooth_service.Advertisement.get_path)
    * [GetAll](#reachy_mini.daemon.app.services.bluetooth.bluetooth_service.Advertisement.GetAll)
    * [Release](#reachy_mini.daemon.app.services.bluetooth.bluetooth_service.Advertisement.Release)
  * [Characteristic](#reachy_mini.daemon.app.services.bluetooth.bluetooth_service.Characteristic)
    * [\_\_init\_\_](#reachy_mini.daemon.app.services.bluetooth.bluetooth_service.Characteristic.__init__)
    * [get\_properties](#reachy_mini.daemon.app.services.bluetooth.bluetooth_service.Characteristic.get_properties)
    * [get\_path](#reachy_mini.daemon.app.services.bluetooth.bluetooth_service.Characteristic.get_path)
    * [GetAll](#reachy_mini.daemon.app.services.bluetooth.bluetooth_service.Characteristic.GetAll)
    * [ReadValue](#reachy_mini.daemon.app.services.bluetooth.bluetooth_service.Characteristic.ReadValue)
    * [WriteValue](#reachy_mini.daemon.app.services.bluetooth.bluetooth_service.Characteristic.WriteValue)
  * [CommandCharacteristic](#reachy_mini.daemon.app.services.bluetooth.bluetooth_service.CommandCharacteristic)
    * [\_\_init\_\_](#reachy_mini.daemon.app.services.bluetooth.bluetooth_service.CommandCharacteristic.__init__)
    * [WriteValue](#reachy_mini.daemon.app.services.bluetooth.bluetooth_service.CommandCharacteristic.WriteValue)
  * [ResponseCharacteristic](#reachy_mini.daemon.app.services.bluetooth.bluetooth_service.ResponseCharacteristic)
    * [\_\_init\_\_](#reachy_mini.daemon.app.services.bluetooth.bluetooth_service.ResponseCharacteristic.__init__)
  * [Service](#reachy_mini.daemon.app.services.bluetooth.bluetooth_service.Service)
    * [\_\_init\_\_](#reachy_mini.daemon.app.services.bluetooth.bluetooth_service.Service.__init__)
    * [get\_properties](#reachy_mini.daemon.app.services.bluetooth.bluetooth_service.Service.get_properties)
    * [get\_path](#reachy_mini.daemon.app.services.bluetooth.bluetooth_service.Service.get_path)
    * [add\_characteristic](#reachy_mini.daemon.app.services.bluetooth.bluetooth_service.Service.add_characteristic)
    * [GetAll](#reachy_mini.daemon.app.services.bluetooth.bluetooth_service.Service.GetAll)
  * [Application](#reachy_mini.daemon.app.services.bluetooth.bluetooth_service.Application)
    * [\_\_init\_\_](#reachy_mini.daemon.app.services.bluetooth.bluetooth_service.Application.__init__)
    * [get\_path](#reachy_mini.daemon.app.services.bluetooth.bluetooth_service.Application.get_path)
    * [GetManagedObjects](#reachy_mini.daemon.app.services.bluetooth.bluetooth_service.Application.GetManagedObjects)
  * [BluetoothCommandService](#reachy_mini.daemon.app.services.bluetooth.bluetooth_service.BluetoothCommandService)
    * [\_\_init\_\_](#reachy_mini.daemon.app.services.bluetooth.bluetooth_service.BluetoothCommandService.__init__)
    * [start](#reachy_mini.daemon.app.services.bluetooth.bluetooth_service.BluetoothCommandService.start)
    * [run](#reachy_mini.daemon.app.services.bluetooth.bluetooth_service.BluetoothCommandService.run)
  * [main](#reachy_mini.daemon.app.services.bluetooth.bluetooth_service.main)
* [reachy\_mini.daemon.app.dependencies](#reachy_mini.daemon.app.dependencies)
  * [get\_daemon](#reachy_mini.daemon.app.dependencies.get_daemon)
  * [get\_backend](#reachy_mini.daemon.app.dependencies.get_backend)
  * [get\_app\_manager](#reachy_mini.daemon.app.dependencies.get_app_manager)
  * [ws\_get\_backend](#reachy_mini.daemon.app.dependencies.ws_get_backend)
* [reachy\_mini.daemon.app.main](#reachy_mini.daemon.app.main)
  * [Args](#reachy_mini.daemon.app.main.Args)
  * [create\_app](#reachy_mini.daemon.app.main.create_app)
  * [run\_app](#reachy_mini.daemon.app.main.run_app)
  * [main](#reachy_mini.daemon.app.main.main)
* [reachy\_mini.daemon.app](#reachy_mini.daemon.app)
* [reachy\_mini.daemon.app.bg\_job\_register](#reachy_mini.daemon.app.bg_job_register)
  * [JobStatus](#reachy_mini.daemon.app.bg_job_register.JobStatus)
  * [JobInfo](#reachy_mini.daemon.app.bg_job_register.JobInfo)
  * [JobHandler](#reachy_mini.daemon.app.bg_job_register.JobHandler)
  * [run\_command](#reachy_mini.daemon.app.bg_job_register.run_command)
  * [get\_info](#reachy_mini.daemon.app.bg_job_register.get_info)
  * [ws\_poll\_info](#reachy_mini.daemon.app.bg_job_register.ws_poll_info)
* [reachy\_mini.daemon.app.models](#reachy_mini.daemon.app.models)
  * [Matrix4x4Pose](#reachy_mini.daemon.app.models.Matrix4x4Pose)
    * [from\_pose\_array](#reachy_mini.daemon.app.models.Matrix4x4Pose.from_pose_array)
    * [to\_pose\_array](#reachy_mini.daemon.app.models.Matrix4x4Pose.to_pose_array)
  * [XYZRPYPose](#reachy_mini.daemon.app.models.XYZRPYPose)
    * [from\_pose\_array](#reachy_mini.daemon.app.models.XYZRPYPose.from_pose_array)
    * [to\_pose\_array](#reachy_mini.daemon.app.models.XYZRPYPose.to_pose_array)
  * [as\_any\_pose](#reachy_mini.daemon.app.models.as_any_pose)
  * [FullBodyTarget](#reachy_mini.daemon.app.models.FullBodyTarget)
  * [FullState](#reachy_mini.daemon.app.models.FullState)
* [reachy\_mini.daemon.backend.abstract](#reachy_mini.daemon.backend.abstract)
  * [MotorControlMode](#reachy_mini.daemon.backend.abstract.MotorControlMode)
    * [Enabled](#reachy_mini.daemon.backend.abstract.MotorControlMode.Enabled)
    * [Disabled](#reachy_mini.daemon.backend.abstract.MotorControlMode.Disabled)
    * [GravityCompensation](#reachy_mini.daemon.backend.abstract.MotorControlMode.GravityCompensation)
  * [Backend](#reachy_mini.daemon.backend.abstract.Backend)
    * [\_\_init\_\_](#reachy_mini.daemon.backend.abstract.Backend.__init__)
    * [wrapped\_run](#reachy_mini.daemon.backend.abstract.Backend.wrapped_run)
    * [run](#reachy_mini.daemon.backend.abstract.Backend.run)
    * [close](#reachy_mini.daemon.backend.abstract.Backend.close)
    * [get\_status](#reachy_mini.daemon.backend.abstract.Backend.get_status)
    * [set\_joint\_positions\_publisher](#reachy_mini.daemon.backend.abstract.Backend.set_joint_positions_publisher)
    * [set\_pose\_publisher](#reachy_mini.daemon.backend.abstract.Backend.set_pose_publisher)
    * [update\_target\_head\_joints\_from\_ik](#reachy_mini.daemon.backend.abstract.Backend.update_target_head_joints_from_ik)
    * [set\_target\_head\_pose](#reachy_mini.daemon.backend.abstract.Backend.set_target_head_pose)
    * [set\_target\_body\_yaw](#reachy_mini.daemon.backend.abstract.Backend.set_target_body_yaw)
    * [set\_target\_head\_joint\_positions](#reachy_mini.daemon.backend.abstract.Backend.set_target_head_joint_positions)
    * [set\_target](#reachy_mini.daemon.backend.abstract.Backend.set_target)
    * [set\_target\_antenna\_joint\_positions](#reachy_mini.daemon.backend.abstract.Backend.set_target_antenna_joint_positions)
    * [set\_target\_head\_joint\_current](#reachy_mini.daemon.backend.abstract.Backend.set_target_head_joint_current)
    * [play\_move](#reachy_mini.daemon.backend.abstract.Backend.play_move)
    * [goto\_target](#reachy_mini.daemon.backend.abstract.Backend.goto_target)
    * [goto\_joint\_positions](#reachy_mini.daemon.backend.abstract.Backend.goto_joint_positions)
    * [set\_recording\_publisher](#reachy_mini.daemon.backend.abstract.Backend.set_recording_publisher)
    * [append\_record](#reachy_mini.daemon.backend.abstract.Backend.append_record)
    * [start\_recording](#reachy_mini.daemon.backend.abstract.Backend.start_recording)
    * [stop\_recording](#reachy_mini.daemon.backend.abstract.Backend.stop_recording)
    * [get\_present\_head\_joint\_positions](#reachy_mini.daemon.backend.abstract.Backend.get_present_head_joint_positions)
    * [get\_present\_body\_yaw](#reachy_mini.daemon.backend.abstract.Backend.get_present_body_yaw)
    * [get\_present\_head\_pose](#reachy_mini.daemon.backend.abstract.Backend.get_present_head_pose)
    * [get\_current\_head\_pose](#reachy_mini.daemon.backend.abstract.Backend.get_current_head_pose)
    * [get\_present\_antenna\_joint\_positions](#reachy_mini.daemon.backend.abstract.Backend.get_present_antenna_joint_positions)
    * [update\_head\_kinematics\_model](#reachy_mini.daemon.backend.abstract.Backend.update_head_kinematics_model)
    * [set\_automatic\_body\_yaw](#reachy_mini.daemon.backend.abstract.Backend.set_automatic_body_yaw)
    * [get\_urdf](#reachy_mini.daemon.backend.abstract.Backend.get_urdf)
    * [play\_sound](#reachy_mini.daemon.backend.abstract.Backend.play_sound)
    * [wake\_up](#reachy_mini.daemon.backend.abstract.Backend.wake_up)
    * [goto\_sleep](#reachy_mini.daemon.backend.abstract.Backend.goto_sleep)
    * [get\_motor\_control\_mode](#reachy_mini.daemon.backend.abstract.Backend.get_motor_control_mode)
    * [set\_motor\_control\_mode](#reachy_mini.daemon.backend.abstract.Backend.set_motor_control_mode)
    * [get\_present\_passive\_joint\_positions](#reachy_mini.daemon.backend.abstract.Backend.get_present_passive_joint_positions)
* [reachy\_mini.daemon.backend.mujoco.utils](#reachy_mini.daemon.backend.mujoco.utils)
  * [get\_homogeneous\_matrix\_from\_euler](#reachy_mini.daemon.backend.mujoco.utils.get_homogeneous_matrix_from_euler)
  * [get\_joint\_qpos](#reachy_mini.daemon.backend.mujoco.utils.get_joint_qpos)
  * [get\_joint\_id\_from\_name](#reachy_mini.daemon.backend.mujoco.utils.get_joint_id_from_name)
  * [get\_joint\_addr\_from\_name](#reachy_mini.daemon.backend.mujoco.utils.get_joint_addr_from_name)
  * [get\_actuator\_names](#reachy_mini.daemon.backend.mujoco.utils.get_actuator_names)
* [reachy\_mini.daemon.backend.mujoco.backend](#reachy_mini.daemon.backend.mujoco.backend)
  * [MujocoBackend](#reachy_mini.daemon.backend.mujoco.backend.MujocoBackend)
    * [\_\_init\_\_](#reachy_mini.daemon.backend.mujoco.backend.MujocoBackend.__init__)
    * [rendering\_loop](#reachy_mini.daemon.backend.mujoco.backend.MujocoBackend.rendering_loop)
    * [run](#reachy_mini.daemon.backend.mujoco.backend.MujocoBackend.run)
    * [get\_mj\_present\_head\_pose](#reachy_mini.daemon.backend.mujoco.backend.MujocoBackend.get_mj_present_head_pose)
    * [close](#reachy_mini.daemon.backend.mujoco.backend.MujocoBackend.close)
    * [get\_status](#reachy_mini.daemon.backend.mujoco.backend.MujocoBackend.get_status)
    * [get\_present\_head\_joint\_positions](#reachy_mini.daemon.backend.mujoco.backend.MujocoBackend.get_present_head_joint_positions)
    * [get\_present\_antenna\_joint\_positions](#reachy_mini.daemon.backend.mujoco.backend.MujocoBackend.get_present_antenna_joint_positions)
    * [get\_motor\_control\_mode](#reachy_mini.daemon.backend.mujoco.backend.MujocoBackend.get_motor_control_mode)
    * [set\_motor\_control\_mode](#reachy_mini.daemon.backend.mujoco.backend.MujocoBackend.set_motor_control_mode)
  * [MujocoBackendStatus](#reachy_mini.daemon.backend.mujoco.backend.MujocoBackendStatus)
* [reachy\_mini.daemon.backend.mujoco.video\_udp](#reachy_mini.daemon.backend.mujoco.video_udp)
  * [UDPJPEGFrameSender](#reachy_mini.daemon.backend.mujoco.video_udp.UDPJPEGFrameSender)
    * [\_\_init\_\_](#reachy_mini.daemon.backend.mujoco.video_udp.UDPJPEGFrameSender.__init__)
    * [send\_frame](#reachy_mini.daemon.backend.mujoco.video_udp.UDPJPEGFrameSender.send_frame)
* [reachy\_mini.daemon.backend.mujoco](#reachy_mini.daemon.backend.mujoco)
* [reachy\_mini.daemon.backend.robot.backend](#reachy_mini.daemon.backend.robot.backend)
  * [RobotBackend](#reachy_mini.daemon.backend.robot.backend.RobotBackend)
    * [\_\_init\_\_](#reachy_mini.daemon.backend.robot.backend.RobotBackend.__init__)
    * [run](#reachy_mini.daemon.backend.robot.backend.RobotBackend.run)
    * [close](#reachy_mini.daemon.backend.robot.backend.RobotBackend.close)
    * [get\_status](#reachy_mini.daemon.backend.robot.backend.RobotBackend.get_status)
    * [enable\_motors](#reachy_mini.daemon.backend.robot.backend.RobotBackend.enable_motors)
    * [disable\_motors](#reachy_mini.daemon.backend.robot.backend.RobotBackend.disable_motors)
    * [set\_head\_operation\_mode](#reachy_mini.daemon.backend.robot.backend.RobotBackend.set_head_operation_mode)
    * [set\_antennas\_operation\_mode](#reachy_mini.daemon.backend.robot.backend.RobotBackend.set_antennas_operation_mode)
    * [get\_all\_joint\_positions](#reachy_mini.daemon.backend.robot.backend.RobotBackend.get_all_joint_positions)
    * [get\_present\_head\_joint\_positions](#reachy_mini.daemon.backend.robot.backend.RobotBackend.get_present_head_joint_positions)
    * [get\_present\_antenna\_joint\_positions](#reachy_mini.daemon.backend.robot.backend.RobotBackend.get_present_antenna_joint_positions)
    * [compensate\_head\_gravity](#reachy_mini.daemon.backend.robot.backend.RobotBackend.compensate_head_gravity)
    * [get\_motor\_control\_mode](#reachy_mini.daemon.backend.robot.backend.RobotBackend.get_motor_control_mode)
    * [set\_motor\_control\_mode](#reachy_mini.daemon.backend.robot.backend.RobotBackend.set_motor_control_mode)
    * [read\_hardware\_errors](#reachy_mini.daemon.backend.robot.backend.RobotBackend.read_hardware_errors)
  * [RobotBackendStatus](#reachy_mini.daemon.backend.robot.backend.RobotBackendStatus)
* [reachy\_mini.daemon.backend.robot](#reachy_mini.daemon.backend.robot)
* [reachy\_mini.daemon.backend](#reachy_mini.daemon.backend)
* [reachy\_mini.daemon.daemon](#reachy_mini.daemon.daemon)
  * [Daemon](#reachy_mini.daemon.daemon.Daemon)
    * [\_\_init\_\_](#reachy_mini.daemon.daemon.Daemon.__init__)
    * [start](#reachy_mini.daemon.daemon.Daemon.start)
    * [stop](#reachy_mini.daemon.daemon.Daemon.stop)
    * [restart](#reachy_mini.daemon.daemon.Daemon.restart)
    * [status](#reachy_mini.daemon.daemon.Daemon.status)
    * [run4ever](#reachy_mini.daemon.daemon.Daemon.run4ever)
  * [DaemonState](#reachy_mini.daemon.daemon.DaemonState)
  * [DaemonStatus](#reachy_mini.daemon.daemon.DaemonStatus)
* [reachy\_mini.daemon](#reachy_mini.daemon)
* [reachy\_mini.motion.move](#reachy_mini.motion.move)
  * [Move](#reachy_mini.motion.move.Move)
    * [duration](#reachy_mini.motion.move.Move.duration)
    * [evaluate](#reachy_mini.motion.move.Move.evaluate)
* [reachy\_mini.motion.goto](#reachy_mini.motion.goto)
  * [GotoMove](#reachy_mini.motion.goto.GotoMove)
    * [\_\_init\_\_](#reachy_mini.motion.goto.GotoMove.__init__)
    * [duration](#reachy_mini.motion.goto.GotoMove.duration)
    * [evaluate](#reachy_mini.motion.goto.GotoMove.evaluate)
* [reachy\_mini.motion.recorded\_move](#reachy_mini.motion.recorded_move)
  * [lerp](#reachy_mini.motion.recorded_move.lerp)
  * [RecordedMove](#reachy_mini.motion.recorded_move.RecordedMove)
    * [\_\_init\_\_](#reachy_mini.motion.recorded_move.RecordedMove.__init__)
    * [duration](#reachy_mini.motion.recorded_move.RecordedMove.duration)
    * [evaluate](#reachy_mini.motion.recorded_move.RecordedMove.evaluate)
  * [RecordedMoves](#reachy_mini.motion.recorded_move.RecordedMoves)
    * [\_\_init\_\_](#reachy_mini.motion.recorded_move.RecordedMoves.__init__)
    * [process](#reachy_mini.motion.recorded_move.RecordedMoves.process)
    * [get](#reachy_mini.motion.recorded_move.RecordedMoves.get)
    * [list\_moves](#reachy_mini.motion.recorded_move.RecordedMoves.list_moves)
* [reachy\_mini.motion](#reachy_mini.motion)

<a id="reachy_mini"></a>

# reachy\_mini

Reachy Mini SDK.

<a id="reachy_mini.reachy_mini"></a>

# reachy\_mini.reachy\_mini

Reachy Mini class for controlling a simulated or real Reachy Mini robot.

This class provides methods to control the head and antennas of the Reachy Mini robot,
set their target positions, and perform various behaviors such as waking up and going to sleep.

It also includes methods for multimedia interactions like playing sounds and looking at specific points in the image frame or world coordinates.

<a id="reachy_mini.reachy_mini.ReachyMini"></a>

## ReachyMini Objects

```python
class ReachyMini()
```

Reachy Mini class for controlling a simulated or real Reachy Mini robot.

**Arguments**:

- `localhost_only` _bool_ - If True, will only connect to localhost daemons, defaults to True.
- `spawn_daemon` _bool_ - If True, will spawn a daemon to control the robot, defaults to False.
- `use_sim` _bool_ - If True and spawn_daemon is True, will spawn a simulated robot, defaults to True.

<a id="reachy_mini.reachy_mini.ReachyMini.__init__"></a>

#### \_\_init\_\_

```python
def __init__(localhost_only: bool = True,
             spawn_daemon: bool = False,
             use_sim: bool = False,
             timeout: float = 5.0,
             automatic_body_yaw: bool = False,
             log_level: str = "INFO",
             media_backend: str = "default") -> None
```

Initialize the Reachy Mini robot.

**Arguments**:

- `localhost_only` _bool_ - If True, will only connect to localhost daemons, defaults to True.
- `spawn_daemon` _bool_ - If True, will spawn a daemon to control the robot, defaults to False.
- `use_sim` _bool_ - If True and spawn_daemon is True, will spawn a simulated robot, defaults to True.
- `timeout` _float_ - Timeout for the client connection, defaults to 5.0 seconds.
- `automatic_body_yaw` _bool_ - If True, the body yaw will be used to compute the IK and FK. Default is False.
- `log_level` _str_ - Logging level, defaults to "INFO".
- `media_backend` _str_ - Media backend to use, either "default" (OpenCV) or "gstreamer", defaults to "default".
  
  It will try to connect to the daemon, and if it fails, it will raise an exception.

<a id="reachy_mini.reachy_mini.ReachyMini.__del__"></a>

#### \_\_del\_\_

```python
def __del__() -> None
```

Destroy the Reachy Mini instance.

The client is disconnected explicitly to avoid a thread pending issue.

<a id="reachy_mini.reachy_mini.ReachyMini.__enter__"></a>

#### \_\_enter\_\_

```python
def __enter__() -> "ReachyMini"
```

Context manager entry point for Reachy Mini.

<a id="reachy_mini.reachy_mini.ReachyMini.__exit__"></a>

#### \_\_exit\_\_

```python
def __exit__(exc_type, exc_value, traceback) -> None
```

Context manager exit point for Reachy Mini.

<a id="reachy_mini.reachy_mini.ReachyMini.media"></a>

#### media

```python
@property
def media() -> MediaManager
```

Expose the MediaManager instance used by ReachyMini.

<a id="reachy_mini.reachy_mini.ReachyMini.set_target"></a>

#### set\_target

```python
def set_target(head: Optional[npt.NDArray[np.float64]] = None,
               antennas: Optional[Union[npt.NDArray[np.float64],
                                        List[float]]] = None,
               body_yaw: Optional[float] = None) -> None
```

Set the target pose of the head and/or the target position of the antennas.

**Arguments**:

- `head` _Optional[np.ndarray]_ - 4x4 pose matrix representing the head pose.
- `antennas` _Optional[Union[np.ndarray, List[float]]]_ - 1D array with two elements representing the angles of the antennas in radians.
- `body_yaw` _Optional[float]_ - Body yaw angle in radians.
  

**Raises**:

- `ValueError` - If neither head nor antennas are provided, or if the shape of head is not (4, 4), or if antennas is not a 1D array with two elements.

<a id="reachy_mini.reachy_mini.ReachyMini.goto_target"></a>

#### goto\_target

```python
def goto_target(
        head: Optional[npt.NDArray[np.float64]] = None,
        antennas: Optional[Union[npt.NDArray[np.float64], List[float]]] = None,
        duration: float = 0.5,
        method: InterpolationTechnique = InterpolationTechnique.MIN_JERK,
        body_yaw: float | None = 0.0) -> None
```

Go to a target head pose and/or antennas position using task space interpolation, in "duration" seconds.

**Arguments**:

- `head` _Optional[np.ndarray]_ - 4x4 pose matrix representing the target head pose.
- `antennas` _Optional[Union[np.ndarray, List[float]]]_ - 1D array with two elements representing the angles of the antennas in radians.
- `duration` _float_ - Duration of the movement in seconds.
- `method` _InterpolationTechnique_ - Interpolation method to use ("linear", "minjerk", "ease", "cartoon"). Default is "minjerk".
- `body_yaw` _float | None_ - Body yaw angle in radians. Use None to keep the current yaw.
  

**Raises**:

- `ValueError` - If neither head nor antennas are provided, or if duration is not positive.

<a id="reachy_mini.reachy_mini.ReachyMini.wake_up"></a>

#### wake\_up

```python
def wake_up() -> None
```

Wake up the robot - go to the initial head position and play the wake up emote and sound.

<a id="reachy_mini.reachy_mini.ReachyMini.goto_sleep"></a>

#### goto\_sleep

```python
def goto_sleep() -> None
```

Put the robot to sleep by moving the head and antennas to a predefined sleep position.

<a id="reachy_mini.reachy_mini.ReachyMini.look_at_image"></a>

#### look\_at\_image

```python
def look_at_image(u: int,
                  v: int,
                  duration: float = 1.0,
                  perform_movement: bool = True) -> npt.NDArray[np.float64]
```

Make the robot head look at a point defined by a pixel position (u,v).

# TODO image of reachy mini coordinate system

**Arguments**:

- `u` _int_ - Horizontal coordinate in image frame.
- `v` _int_ - Vertical coordinate in image frame.
- `duration` _float_ - Duration of the movement in seconds. If 0, the head will snap to the position immediately.
- `perform_movement` _bool_ - If True, perform the movement. If False, only calculate and return the pose.
  

**Returns**:

- `np.ndarray` - The calculated head pose as a 4x4 matrix.
  

**Raises**:

- `ValueError` - If duration is negative.

<a id="reachy_mini.reachy_mini.ReachyMini.look_at_world"></a>

#### look\_at\_world

```python
def look_at_world(x: float,
                  y: float,
                  z: float,
                  duration: float = 1.0,
                  perform_movement: bool = True) -> npt.NDArray[np.float64]
```

Look at a specific point in 3D space in Reachy Mini's reference frame.

TODO include image of reachy mini coordinate system

**Arguments**:

- `x` _float_ - X coordinate in meters.
- `y` _float_ - Y coordinate in meters.
- `z` _float_ - Z coordinate in meters.
- `duration` _float_ - Duration of the movement in seconds. If 0, the head will snap to the position immediately.
- `perform_movement` _bool_ - If True, perform the movement. If False, only calculate and return the pose.
  

**Returns**:

- `np.ndarray` - The calculated head pose as a 4x4 matrix.
  

**Raises**:

- `ValueError` - If duration is negative.

<a id="reachy_mini.reachy_mini.ReachyMini.get_current_joint_positions"></a>

#### get\_current\_joint\_positions

```python
def get_current_joint_positions() -> tuple[list[float], list[float]]
```

Get the current joint positions of the head and antennas.

Get the current joint positions of the head and antennas (in rad)

**Returns**:

- `tuple` - A tuple containing two lists:
  - List of head joint positions (rad) (length 7).
  - List of antennas joint positions (rad) (length 2).

<a id="reachy_mini.reachy_mini.ReachyMini.get_present_antenna_joint_positions"></a>

#### get\_present\_antenna\_joint\_positions

```python
def get_present_antenna_joint_positions() -> list[float]
```

Get the present joint positions of the antennas.

Get the present joint positions of the antennas (in rad)

**Returns**:

- `list` - A list of antennas joint positions (rad) (length 2).

<a id="reachy_mini.reachy_mini.ReachyMini.get_current_head_pose"></a>

#### get\_current\_head\_pose

```python
def get_current_head_pose() -> npt.NDArray[np.float64]
```

Get the current head pose as a 4x4 matrix.

Get the current head pose as a 4x4 matrix.

**Returns**:

- `np.ndarray` - A 4x4 matrix representing the current head pose.

<a id="reachy_mini.reachy_mini.ReachyMini.set_target_head_pose"></a>

#### set\_target\_head\_pose

```python
def set_target_head_pose(pose: npt.NDArray[np.float64]) -> None
```

Set the head pose to a specific 4x4 matrix.

**Arguments**:

- `pose` _np.ndarray_ - A 4x4 matrix representing the desired head pose.
- `body_yaw` _float_ - The yaw angle of the body, used to adjust the head pose.
  

**Raises**:

- `ValueError` - If the shape of the pose is not (4, 4).

<a id="reachy_mini.reachy_mini.ReachyMini.set_target_antenna_joint_positions"></a>

#### set\_target\_antenna\_joint\_positions

```python
def set_target_antenna_joint_positions(antennas: List[float]) -> None
```

Set the target joint positions of the antennas.

<a id="reachy_mini.reachy_mini.ReachyMini.set_target_body_yaw"></a>

#### set\_target\_body\_yaw

```python
def set_target_body_yaw(body_yaw: float) -> None
```

Set the target body yaw.

**Arguments**:

- `body_yaw` _float_ - The yaw angle of the body in radians.

<a id="reachy_mini.reachy_mini.ReachyMini.start_recording"></a>

#### start\_recording

```python
def start_recording() -> None
```

Start recording data.

<a id="reachy_mini.reachy_mini.ReachyMini.stop_recording"></a>

#### stop\_recording

```python
def stop_recording(
) -> Optional[List[Dict[str, float | List[float] | List[List[float]]]]]
```

Stop recording data and return the recorded data.

<a id="reachy_mini.reachy_mini.ReachyMini.enable_motors"></a>

#### enable\_motors

```python
def enable_motors() -> None
```

Enable the motors.

<a id="reachy_mini.reachy_mini.ReachyMini.disable_motors"></a>

#### disable\_motors

```python
def disable_motors() -> None
```

Disable the motors.

<a id="reachy_mini.reachy_mini.ReachyMini.enable_gravity_compensation"></a>

#### enable\_gravity\_compensation

```python
def enable_gravity_compensation() -> None
```

Enable gravity compensation for the head motors.

<a id="reachy_mini.reachy_mini.ReachyMini.disable_gravity_compensation"></a>

#### disable\_gravity\_compensation

```python
def disable_gravity_compensation() -> None
```

Disable gravity compensation for the head motors.

<a id="reachy_mini.reachy_mini.ReachyMini.set_automatic_body_yaw"></a>

#### set\_automatic\_body\_yaw

```python
def set_automatic_body_yaw(body_yaw: float) -> None
```

Set the automatic body yaw.

**Arguments**:

- `body_yaw` _float_ - The yaw angle of the body in radians.

<a id="reachy_mini.reachy_mini.ReachyMini.async_play_move"></a>

#### async\_play\_move

```python
async def async_play_move(move: Move,
                          play_frequency: float = 100.0,
                          initial_goto_duration: float = 0.0) -> None
```

Asynchronously play a Move.

**Arguments**:

- `move` _Move_ - The Move object to be played.
- `play_frequency` _float_ - The frequency at which to evaluate the move (in Hz).
- `initial_goto_duration` _float_ - Duration for the initial goto to the starting position of the move (in seconds). If 0, no initial goto is performed.

<a id="reachy_mini.kinematics.analytical_kinematics"></a>

# reachy\_mini.kinematics.analytical\_kinematics

An analytical kinematics engine for Reachy Mini, using Rust bindings.

The inverse kinematics use an analytical method, while the forward kinematics
use a numerical method (Newton).

<a id="reachy_mini.kinematics.analytical_kinematics.AnalyticalKinematics"></a>

## AnalyticalKinematics Objects

```python
class AnalyticalKinematics()
```

Reachy Mini Analytical Kinematics class, implemented in Rust with python bindings.

<a id="reachy_mini.kinematics.analytical_kinematics.AnalyticalKinematics.__init__"></a>

#### \_\_init\_\_

```python
def __init__() -> None
```

Initialize.

<a id="reachy_mini.kinematics.analytical_kinematics.AnalyticalKinematics.ik"></a>

#### ik

```python
def ik(pose: Annotated[NDArray[np.float64], (4, 4)],
       body_yaw: float = 0.0,
       check_collision: bool = False,
       no_iterations: int = 0) -> Annotated[NDArray[np.float64], (7, )]
```

Compute the inverse kinematics for a given head pose.

check_collision and no_iterations are not used by AnalyticalKinematics. We keep them for compatibility with the other kinematics engines

<a id="reachy_mini.kinematics.analytical_kinematics.AnalyticalKinematics.fk"></a>

#### fk

```python
def fk(joint_angles: Annotated[NDArray[np.float64], (7, )],
       check_collision: bool = False,
       no_iterations: int = 3) -> Annotated[NDArray[np.float64], (4, 4)]
```

Compute the forward kinematics for a given set of joint angles.

check_collision is not used by AnalyticalKinematics.

<a id="reachy_mini.kinematics.placo_kinematics"></a>

# reachy\_mini.kinematics.placo\_kinematics

Placo Kinematics for Reachy Mini.

This module provides the PlacoKinematics class for performing inverse and forward kinematics based on the Reachy Mini robot URDF using the Placo library.

<a id="reachy_mini.kinematics.placo_kinematics.PlacoKinematics"></a>

## PlacoKinematics Objects

```python
class PlacoKinematics()
```

Placo Kinematics class for Reachy Mini.

This class provides methods for inverse and forward kinematics using the Placo library and a URDF model of the Reachy Mini robot.

<a id="reachy_mini.kinematics.placo_kinematics.PlacoKinematics.__init__"></a>

#### \_\_init\_\_

```python
def __init__(urdf_path: str,
             dt: float = 0.02,
             automatic_body_yaw: bool = False,
             check_collision: bool = False,
             log_level: str = "INFO") -> None
```

Initialize the PlacoKinematics class.

**Arguments**:

- `urdf_path` _str_ - Path to the URDF file of the Reachy Mini robot.
- `dt` _float_ - Time step for the kinematics solver. Default is 0.02 seconds.
- `automatic_body_yaw` _bool_ - If True, the body yaw will be used to compute the IK and FK. Default is False.
- `check_collision` _bool_ - If True, checks for collisions after solving IK. (default: False)
- `log_level` _str_ - Logging level for the kinematics computations.

<a id="reachy_mini.kinematics.placo_kinematics.PlacoKinematics.ik"></a>

#### ik

```python
def ik(
    pose: npt.NDArray[np.float64],
    body_yaw: float = 0.0,
    no_iterations: int = 2
) -> Annotated[npt.NDArray[np.float64], (7, )] | None
```

Compute the inverse kinematics for the head for a given pose.

**Arguments**:

- `pose` _np.ndarray_ - A 4x4 homogeneous transformation matrix
  representing the desired position and orientation of the head.
- `body_yaw` _float_ - Body yaw angle in radians.
- `no_iterations` _int_ - Number of iterations to perform (default: 2). The higher the value, the more accurate the solution.
  

**Returns**:

- `List[float]` - A list of joint angles for the head.

<a id="reachy_mini.kinematics.placo_kinematics.PlacoKinematics.fk"></a>

#### fk

```python
def fk(joints_angles: Annotated[npt.NDArray[np.float64], (7, )],
       no_iterations: int = 2) -> Optional[npt.NDArray[np.float64]]
```

Compute the forward kinematics for the head given joint angles.

**Arguments**:

- `joints_angles` _List[float]_ - A list of joint angles for the head.
- `no_iterations` _int_ - The number of iterations to use for the FK solver. (default: 2), the higher the more accurate the result.
  

**Returns**:

- `np.ndarray` - A 4x4 homogeneous transformation matrix

<a id="reachy_mini.kinematics.placo_kinematics.PlacoKinematics.config_collision_model"></a>

#### config\_collision\_model

```python
def config_collision_model() -> None
```

Configure the collision model for the robot.

Add collision pairs between the torso and the head colliders.

<a id="reachy_mini.kinematics.placo_kinematics.PlacoKinematics.compute_collision"></a>

#### compute\_collision

```python
def compute_collision(margin: float = 0.005) -> bool
```

Compute the collision between the robot and the environment.

**Arguments**:

- `margin` _float_ - The margin to consider for collision detection (default: 5mm).
  

**Returns**:

  True if there is a collision, False otherwise.

<a id="reachy_mini.kinematics.placo_kinematics.PlacoKinematics.compute_jacobian"></a>

#### compute\_jacobian

```python
def compute_jacobian(
        q: Optional[npt.NDArray[np.float64]] = None
) -> npt.NDArray[np.float64]
```

Compute the Jacobian of the head frame with respect to the actuated DoFs.

The jacobian in local world aligned.

**Arguments**:

- `q` _np.ndarray, optional_ - Joint angles of the robot. If None, uses the current state of the robot. (default: None)
  

**Returns**:

- `np.ndarray` - The Jacobian matrix.

<a id="reachy_mini.kinematics.placo_kinematics.PlacoKinematics.compute_gravity_torque"></a>

#### compute\_gravity\_torque

```python
def compute_gravity_torque(
        q: Optional[npt.NDArray[np.float64]] = None
) -> npt.NDArray[np.float64]
```

Compute the gravity torque vector for the actuated joints of the robot.

This method uses the static gravity compensation torques from the robot's dictionary.

**Arguments**:

- `q` _np.ndarray, optional_ - Joint angles of the robot. If None, uses the current state of the robot. (default: None)
  

**Returns**:

- `np.ndarray` - The gravity torque vector.

<a id="reachy_mini.kinematics.placo_kinematics.PlacoKinematics.set_automatic_body_yaw"></a>

#### set\_automatic\_body\_yaw

```python
def set_automatic_body_yaw(body_yaw: float) -> None
```

Set the automatic body yaw.

**Arguments**:

- `body_yaw` _float_ - The yaw angle of the body.

<a id="reachy_mini.kinematics.placo_kinematics.PlacoKinematics.get_joint"></a>

#### get\_joint

```python
def get_joint(joint_name: str) -> float
```

Get the joint object by its name.

<a id="reachy_mini.kinematics"></a>

# reachy\_mini.kinematics

Try to import kinematics engines, and provide mockup classes if they are not available.

<a id="reachy_mini.kinematics.nn_kinematics"></a>

# reachy\_mini.kinematics.nn\_kinematics

Neural Network based FK/IK.

<a id="reachy_mini.kinematics.nn_kinematics.NNKinematics"></a>

## NNKinematics Objects

```python
class NNKinematics()
```

Neural Network based FK/IK. Fitted from PlacoKinematics data.

<a id="reachy_mini.kinematics.nn_kinematics.NNKinematics.__init__"></a>

#### \_\_init\_\_

```python
def __init__(models_root_path: str)
```

Intialize.

<a id="reachy_mini.kinematics.nn_kinematics.NNKinematics.ik"></a>

#### ik

```python
def ik(pose: Annotated[npt.NDArray[np.float64], (4, 4)],
       body_yaw: float = 0.0,
       check_collision: bool = False,
       no_iterations: int = 0) -> Annotated[npt.NDArray[np.float64], (7, )]
```

check_collision and no_iterations are not used by NNKinematics.

We keep them for compatibility with the other kinematics engines

<a id="reachy_mini.kinematics.nn_kinematics.NNKinematics.fk"></a>

#### fk

```python
def fk(joint_angles: Annotated[npt.NDArray[np.float64], (7, )],
       check_collision: bool = False,
       no_iterations: int = 0) -> Annotated[npt.NDArray[np.float64], (4, 4)]
```

check_collision and no_iterations are not used by NNKinematics.

We keep them for compatibility with the other kinematics engines

<a id="reachy_mini.kinematics.nn_kinematics.OnnxInfer"></a>

## OnnxInfer Objects

```python
class OnnxInfer()
```

Infer an onnx model.

<a id="reachy_mini.kinematics.nn_kinematics.OnnxInfer.__init__"></a>

#### \_\_init\_\_

```python
def __init__(onnx_model_path: str) -> None
```

Initialize.

<a id="reachy_mini.kinematics.nn_kinematics.OnnxInfer.infer"></a>

#### infer

```python
def infer(input: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]
```

Run inference on the input.

<a id="reachy_mini.utils.constants"></a>

# reachy\_mini.utils.constants

Utility constants for the reachy_mini package.

<a id="reachy_mini.utils.wireless_version.utils"></a>

# reachy\_mini.utils.wireless\_version.utils

Utility functions for running shell commands asynchronously with real-time logging.

<a id="reachy_mini.utils.wireless_version.utils.call_logger_wrapper"></a>

#### call\_logger\_wrapper

```python
async def call_logger_wrapper(command: list[str],
                              logger: logging.Logger) -> None
```

Run a command asynchronously, streaming stdout and stderr to logger in real time.

**Arguments**:

- `command` - list or tuple of command arguments (not a string)
- `logger` - logger object with .info and .error methods

<a id="reachy_mini.utils.wireless_version.update"></a>

# reachy\_mini.utils.wireless\_version.update

Module to handle software updates for the Reachy Mini wireless.

<a id="reachy_mini.utils.wireless_version.update.update_reachy_mini"></a>

#### update\_reachy\_mini

```python
async def update_reachy_mini(pre_release: bool,
                             logger: logging.Logger) -> None
```

Perform a software update by upgrading the reachy_mini package and restarting the daemon.

<a id="reachy_mini.utils.wireless_version"></a>

# reachy\_mini.utils.wireless\_version

Utility functions for working with wireless version.

<a id="reachy_mini.utils.wireless_version.update_available"></a>

# reachy\_mini.utils.wireless\_version.update\_available

Check if an update is available for Reachy Mini Wireless.

For now, this only checks if a new version of "reachy_mini" is available on PyPI.

<a id="reachy_mini.utils.wireless_version.update_available.is_update_available"></a>

#### is\_update\_available

```python
def is_update_available(package_name: str, pre_release: bool) -> bool
```

Check if an update is available for the given package.

<a id="reachy_mini.utils.wireless_version.update_available.get_pypi_version"></a>

#### get\_pypi\_version

```python
def get_pypi_version(package_name: str, pre_release: bool) -> semver.Version
```

Get the latest version of a package from PyPI.

<a id="reachy_mini.utils.wireless_version.update_available.get_local_version"></a>

#### get\_local\_version

```python
def get_local_version(package_name: str) -> semver.Version
```

Get the currently installed version of a package.

<a id="reachy_mini.utils.interpolation"></a>

# reachy\_mini.utils.interpolation

Interpolation utilities for Reachy Mini.

<a id="reachy_mini.utils.interpolation.minimum_jerk"></a>

#### minimum\_jerk

```python
def minimum_jerk(
    starting_position: npt.NDArray[np.float64],
    goal_position: npt.NDArray[np.float64],
    duration: float,
    starting_velocity: Optional[npt.NDArray[np.float64]] = None,
    starting_acceleration: Optional[npt.NDArray[np.float64]] = None,
    final_velocity: Optional[npt.NDArray[np.float64]] = None,
    final_acceleration: Optional[npt.NDArray[np.float64]] = None
) -> InterpolationFunc
```

Compute the mimimum jerk interpolation function from starting position to goal position.

<a id="reachy_mini.utils.interpolation.linear_pose_interpolation"></a>

#### linear\_pose\_interpolation

```python
def linear_pose_interpolation(start_pose: npt.NDArray[np.float64],
                              target_pose: npt.NDArray[np.float64],
                              t: float) -> npt.NDArray[np.float64]
```

Linearly interpolate between two poses in 6D space.

<a id="reachy_mini.utils.interpolation.InterpolationTechnique"></a>

## InterpolationTechnique Objects

```python
class InterpolationTechnique(str, Enum)
```

Enumeration of interpolation techniques.

<a id="reachy_mini.utils.interpolation.time_trajectory"></a>

#### time\_trajectory

```python
def time_trajectory(
        t: float,
        method: InterpolationTechnique = InterpolationTechnique.MIN_JERK
) -> float
```

Compute the time trajectory value based on the specified interpolation method.

<a id="reachy_mini.utils.interpolation.delta_angle_between_mat_rot"></a>

#### delta\_angle\_between\_mat\_rot

```python
def delta_angle_between_mat_rot(P: npt.NDArray[np.float64],
                                Q: npt.NDArray[np.float64]) -> float
```

Compute the angle (in radians) between two 3x3 rotation matrices `P` and `Q`.

This is equivalent to the angular distance in axis-angle space.
It is computed via the trace of the relative rotation matrix.

**References**:

  - https://math.stackexchange.com/questions/2113634/comparing-two-rotation-matrices
  - http://www.boris-belousov.net/2016/12/01/quat-dist/
  

**Arguments**:

- `P` - A 3x3 rotation matrix.
- `Q` - Another 3x3 rotation matrix.
  

**Returns**:

  The angle in radians between the two rotations.

<a id="reachy_mini.utils.interpolation.distance_between_poses"></a>

#### distance\_between\_poses

```python
def distance_between_poses(
        pose1: npt.NDArray[np.float64],
        pose2: npt.NDArray[np.float64]) -> Tuple[float, float, float]
```

Compute three types of distance between two 4x4 homogeneous transformation matrices.

The result combines translation (in mm) and rotation (in degrees) using an arbitrary but
emotionally satisfying equivalence: 1 degree ≈ 1 mm.

**Arguments**:

- `pose1` - A 4x4 homogeneous transformation matrix representing the first pose.
- `pose2` - A 4x4 homogeneous transformation matrix representing the second pose.
  

**Returns**:

  A tuple of:
  - translation distance in meters,
  - angular distance in radians,
  - unhinged distance in magic-mm (translation in mm + rotation in degrees).

<a id="reachy_mini.utils.interpolation.compose_world_offset"></a>

#### compose\_world\_offset

```python
def compose_world_offset(
        T_abs: npt.NDArray[np.float64],
        T_off_world: npt.NDArray[np.float64],
        reorthonormalize: bool = False) -> npt.NDArray[np.float64]
```

Compose an absolute world-frame pose with a world-frame offset.

  - translations add in world:       t_final = t_abs + t_off
  - rotations compose in world:      R_final = R_off @ R_abs
This rotates the frame in place (about its own origin) by a rotation
defined in world axes, and shifts it by a world translation.

Parameters
----------
T_abs : (4,4) ndarray
    Absolute pose in world frame.
T_off_world : (4,4) ndarray
    Offset transform specified in world axes (dx,dy,dz in world; dR about world axes).
reorthonormalize : bool
    If True, SVD-orthonormalize the resulting rotation to fight drift.

Returns
-------
T_final : (4,4) ndarray
    Resulting pose in world frame.

<a id="reachy_mini.utils.rerun"></a>

# reachy\_mini.utils.rerun

Rerun logging for Reachy Mini.

This module provides functionality to log the state of the Reachy Mini robot to Rerun,
 a tool for visualizing and debugging robotic systems.

It includes methods to log joint positions, camera images, and other relevant data.

<a id="reachy_mini.utils.rerun.Rerun"></a>

## Rerun Objects

```python
class Rerun()
```

Rerun logging for Reachy Mini.

<a id="reachy_mini.utils.rerun.Rerun.__init__"></a>

#### \_\_init\_\_

```python
def __init__(reachymini: ReachyMini,
             app_id: str = "reachy_mini_rerun",
             spawn: bool = True)
```

Initialize the Rerun logging for Reachy Mini.

**Arguments**:

- `reachymini` _ReachyMini_ - The Reachy Mini instance to log.
- `app_id` _str_ - The application ID for Rerun. Defaults to reachy_mini_daemon.
- `spawn` _bool_ - If True, spawn the Rerun server. Defaults to True.

<a id="reachy_mini.utils.rerun.Rerun.set_absolute_path_to_urdf"></a>

#### set\_absolute\_path\_to\_urdf

```python
def set_absolute_path_to_urdf(urdf_path: str, abs_path: str) -> str
```

Set the absolute paths in the URDF file. Rerun cannot read the "package://" paths.

<a id="reachy_mini.utils.rerun.Rerun.start"></a>

#### start

```python
def start() -> None
```

Start the Rerun logging thread.

<a id="reachy_mini.utils.rerun.Rerun.stop"></a>

#### stop

```python
def stop() -> None
```

Stop the Rerun logging thread.

<a id="reachy_mini.utils.rerun.Rerun.log_camera"></a>

#### log\_camera

```python
def log_camera() -> None
```

Log the camera image to Rerun.

<a id="reachy_mini.utils.rerun.Rerun.log_movements"></a>

#### log\_movements

```python
def log_movements() -> None
```

Log the movement data to Rerun.

<a id="reachy_mini.utils"></a>

# reachy\_mini.utils

Utility functions for Reachy Mini.

These functions provide various utilities such as creating head poses, performing minimum jerk interpolation,
checking if the Reachy Mini daemon is running, and performing linear pose interpolation.

<a id="reachy_mini.utils.create_head_pose"></a>

#### create\_head\_pose

```python
def create_head_pose(x: float = 0,
                     y: float = 0,
                     z: float = 0,
                     roll: float = 0,
                     pitch: float = 0,
                     yaw: float = 0,
                     mm: bool = False,
                     degrees: bool = True) -> npt.NDArray[np.float64]
```

Create a homogeneous transformation matrix representing a pose in 6D space (position and orientation).

**Arguments**:

- `x` _float_ - X coordinate of the position.
- `y` _float_ - Y coordinate of the position.
- `z` _float_ - Z coordinate of the position.
- `roll` _float_ - Roll angle
- `pitch` _float_ - Pitch angle
- `yaw` _float_ - Yaw angle
- `mm` _bool_ - If True, convert position from millimeters to meters.
- `degrees` _bool_ - If True, interpret roll, pitch, and yaw as degrees; otherwise as radians.
  

**Returns**:

- `np.ndarray` - A 4x4 homogeneous transformation matrix representing the pose.

<a id="reachy_mini.utils.parse_urdf_for_kinematics"></a>

# reachy\_mini.utils.parse\_urdf\_for\_kinematics

Generate kinematics data from URDF using Placo as preprocessing.

The analytical kinematics need information from the URDF. This files computes the information and writes it in a .json file.

<a id="reachy_mini.utils.parse_urdf_for_kinematics.get_data"></a>

#### get\_data

```python
def get_data() -> Dict[str, Any]
```

Generate the urdf_kinematics.json file.

<a id="reachy_mini.utils.parse_urdf_for_kinematics.main"></a>

#### main

```python
def main() -> None
```

Generate the urdf_kinematics.json file.

<a id="reachy_mini.apps.utils"></a>

# reachy\_mini.apps.utils

Utility functions for Reachy Mini apps manager.

<a id="reachy_mini.apps.utils.running_command"></a>

#### running\_command

```python
async def running_command(command: list[str], logger: logging.Logger) -> int
```

Run a shell command and stream its output to the provided logger.

<a id="reachy_mini.apps.sources.hf_space"></a>

# reachy\_mini.apps.sources.hf\_space

Hugging Face Spaces app source.

<a id="reachy_mini.apps.sources.hf_space.list_available_apps"></a>

#### list\_available\_apps

```python
async def list_available_apps() -> list[AppInfo]
```

List apps available on Hugging Face Spaces.

<a id="reachy_mini.apps.sources"></a>

# reachy\_mini.apps.sources

Specific app sources.

For the moment, only huggingface spaces is implemented.

<a id="reachy_mini.apps.sources.local_common_venv"></a>

# reachy\_mini.apps.sources.local\_common\_venv

Utilities for local common venv apps source.

<a id="reachy_mini.apps.sources.local_common_venv.list_available_apps"></a>

#### list\_available\_apps

```python
async def list_available_apps() -> list[AppInfo]
```

List apps available from entry points.

<a id="reachy_mini.apps.sources.local_common_venv.install_package"></a>

#### install\_package

```python
async def install_package(app: AppInfo, logger: logging.Logger) -> int
```

Install a package given an AppInfo object, streaming logs.

<a id="reachy_mini.apps.sources.local_common_venv.uninstall_package"></a>

#### uninstall\_package

```python
async def uninstall_package(app_name: str, logger: logging.Logger) -> int
```

Uninstall a package given an app name.

<a id="reachy_mini.apps.manager"></a>

# reachy\_mini.apps.manager

App management for Reachy Mini.

<a id="reachy_mini.apps.manager.AppState"></a>

## AppState Objects

```python
class AppState(str, Enum)
```

Status of a running app.

<a id="reachy_mini.apps.manager.AppStatus"></a>

## AppStatus Objects

```python
class AppStatus(BaseModel)
```

Status of an app.

<a id="reachy_mini.apps.manager.RunningApp"></a>

## RunningApp Objects

```python
@dataclass
class RunningApp()
```

Information about a running app.

<a id="reachy_mini.apps.manager.AppManager"></a>

## AppManager Objects

```python
class AppManager()
```

Manager for Reachy Mini apps.

<a id="reachy_mini.apps.manager.AppManager.__init__"></a>

#### \_\_init\_\_

```python
def __init__() -> None
```

Initialize the AppManager.

<a id="reachy_mini.apps.manager.AppManager.close"></a>

#### close

```python
async def close() -> None
```

Clean up the AppManager, stopping any running app.

<a id="reachy_mini.apps.manager.AppManager.is_app_running"></a>

#### is\_app\_running

```python
def is_app_running() -> bool
```

Check if an app is currently running.

<a id="reachy_mini.apps.manager.AppManager.start_app"></a>

#### start\_app

```python
async def start_app(app_name: str, *args: Any, **kwargs: Any) -> AppStatus
```

Start the app, raises RuntimeError if an app is already running.

<a id="reachy_mini.apps.manager.AppManager.stop_current_app"></a>

#### stop\_current\_app

```python
async def stop_current_app(timeout: float | None = 5.0) -> None
```

Stop the current app.

<a id="reachy_mini.apps.manager.AppManager.restart_current_app"></a>

#### restart\_current\_app

```python
async def restart_current_app() -> AppStatus
```

Restart the current app.

<a id="reachy_mini.apps.manager.AppManager.current_app_status"></a>

#### current\_app\_status

```python
async def current_app_status() -> Optional[AppStatus]
```

Get the current status of the app.

<a id="reachy_mini.apps.manager.AppManager.list_all_available_apps"></a>

#### list\_all\_available\_apps

```python
async def list_all_available_apps() -> list[AppInfo]
```

List available apps (parallel async).

<a id="reachy_mini.apps.manager.AppManager.list_available_apps"></a>

#### list\_available\_apps

```python
async def list_available_apps(source: SourceKind) -> list[AppInfo]
```

List available apps for given source kind.

<a id="reachy_mini.apps.manager.AppManager.install_new_app"></a>

#### install\_new\_app

```python
async def install_new_app(app: AppInfo, logger: logging.Logger) -> None
```

Install a new app by name.

<a id="reachy_mini.apps.manager.AppManager.remove_app"></a>

#### remove\_app

```python
async def remove_app(app_name: str, logger: logging.Logger) -> None
```

Remove an installed app by name.

<a id="reachy_mini.apps.app"></a>

# reachy\_mini.apps.app

Reachy Mini Application Base Class.

This module provides a base class for creating Reachy Mini applications.
It includes methods for running the application, stopping it gracefully,
and creating a new app project with a specified name and path.

It uses Jinja2 templates to generate the necessary files for the app project.

<a id="reachy_mini.apps.app.ReachyMiniApp"></a>

## ReachyMiniApp Objects

```python
class ReachyMiniApp(ABC)
```

Base class for Reachy Mini applications.

<a id="reachy_mini.apps.app.ReachyMiniApp.__init__"></a>

#### \_\_init\_\_

```python
def __init__() -> None
```

Initialize the Reachy Mini app.

<a id="reachy_mini.apps.app.ReachyMiniApp.wrapped_run"></a>

#### wrapped\_run

```python
def wrapped_run(*args: Any, **kwargs: Any) -> None
```

Wrap the run method with Reachy Mini context management.

<a id="reachy_mini.apps.app.ReachyMiniApp.run"></a>

#### run

```python
@abstractmethod
def run(reachy_mini: ReachyMini, stop_event: threading.Event) -> None
```

Run the main logic of the app.

**Arguments**:

- `reachy_mini` _ReachyMini_ - The Reachy Mini instance to interact with.
- `stop_event` _threading.Event_ - An event that can be set to stop the app gracefully.

<a id="reachy_mini.apps.app.ReachyMiniApp.stop"></a>

#### stop

```python
def stop() -> None
```

Stop the app gracefully.

<a id="reachy_mini.apps.app.make_app_project"></a>

#### make\_app\_project

```python
def make_app_project(app_name: str, path: Path) -> None
```

Create a new Reachy Mini app project with the given name at the specified path.

**Arguments**:

- `app_name` _str_ - The name of the app to create.
- `path` _Path_ - The directory where the app project will be created.

<a id="reachy_mini.apps.app.main"></a>

#### main

```python
def main() -> None
```

Run the command line interface to create a new Reachy Mini app project.

<a id="reachy_mini.apps"></a>

# reachy\_mini.apps

Metadata about apps.

<a id="reachy_mini.apps.SourceKind"></a>

## SourceKind Objects

```python
class SourceKind(str, Enum)
```

Kinds of app source.

<a id="reachy_mini.apps.AppInfo"></a>

## AppInfo Objects

```python
@dataclass
class AppInfo()
```

Metadata about an app.

<a id="reachy_mini.media.camera_gstreamer"></a>

# reachy\_mini.media.camera\_gstreamer

GStreamer camera backend.

This module provides an implementation of the CameraBase class using GStreamer.
By default the module directly returns JPEG images as output by the camera.

<a id="reachy_mini.media.camera_gstreamer.GStreamerCamera"></a>

## GStreamerCamera Objects

```python
class GStreamerCamera(CameraBase)
```

Camera implementation using GStreamer.

<a id="reachy_mini.media.camera_gstreamer.GStreamerCamera.__init__"></a>

#### \_\_init\_\_

```python
def __init__(
        log_level: str = "INFO",
        resolution: CameraResolution = CameraResolution.R1280x720) -> None
```

Initialize the GStreamer camera.

<a id="reachy_mini.media.camera_gstreamer.GStreamerCamera.open"></a>

#### open

```python
def open() -> None
```

Open the camera using GStreamer.

<a id="reachy_mini.media.camera_gstreamer.GStreamerCamera.read"></a>

#### read

```python
def read() -> Optional[npt.NDArray[np.uint8]]
```

Read a frame from the camera. Returns the frame or None if error.

**Returns**:

- `Optional[npt.NDArray[np.uint8]]` - The captured BGR frame as a NumPy array, or None if error.

<a id="reachy_mini.media.camera_gstreamer.GStreamerCamera.close"></a>

#### close

```python
def close() -> None
```

Release the camera resource.

<a id="reachy_mini.media.camera_gstreamer.GStreamerCamera.get_arducam_video_device"></a>

#### get\_arducam\_video\_device

```python
def get_arducam_video_device() -> str
```

Use Gst.DeviceMonitor to find the unix camera path /dev/videoX of the Arducam_12MP webcam.

Returns the device path (e.g., '/dev/video2'), or '' if not found.

<a id="reachy_mini.media.webrtc_client_gstreamer"></a>

# reachy\_mini.media.webrtc\_client\_gstreamer

GStreamer WebRTC client implementation.

The class is a client for the webrtc server hosted on the Reachy Mini Wireless robot.

<a id="reachy_mini.media.webrtc_client_gstreamer.GstWebRTCClient"></a>

## GstWebRTCClient Objects

```python
class GstWebRTCClient(CameraBase, AudioBase)
```

GStreamer WebRTC client implementation.

<a id="reachy_mini.media.webrtc_client_gstreamer.GstWebRTCClient.__init__"></a>

#### \_\_init\_\_

```python
def __init__(log_level: str = "INFO",
             peer_id: str = "",
             signaling_host: str = "",
             signaling_port: int = 8443)
```

Initialize the GStreamer WebRTC client.

<a id="reachy_mini.media.webrtc_client_gstreamer.GstWebRTCClient.__del__"></a>

#### \_\_del\_\_

```python
def __del__() -> None
```

Destructor to ensure gstreamer resources are released.

<a id="reachy_mini.media.webrtc_client_gstreamer.GstWebRTCClient.open"></a>

#### open

```python
def open() -> None
```

Open the video stream.

<a id="reachy_mini.media.webrtc_client_gstreamer.GstWebRTCClient.get_audio_sample"></a>

#### get\_audio\_sample

```python
def get_audio_sample() -> Optional[npt.NDArray[np.float32]]
```

Read a sample from the audio card. Returns the sample or None if error.

**Returns**:

- `Optional[npt.NDArray[np.float32]]` - The captured sample in raw format, or None if error.

<a id="reachy_mini.media.webrtc_client_gstreamer.GstWebRTCClient.read"></a>

#### read

```python
def read() -> Optional[npt.NDArray[np.uint8]]
```

Read a frame from the camera. Returns the frame or None if error.

**Returns**:

- `Optional[npt.NDArray[np.uint8]]` - The captured frame in BGR format, or None if error.

<a id="reachy_mini.media.webrtc_client_gstreamer.GstWebRTCClient.close"></a>

#### close

```python
def close() -> None
```

Stop the pipeline.

<a id="reachy_mini.media.webrtc_client_gstreamer.GstWebRTCClient.start_recording"></a>

#### start\_recording

```python
def start_recording() -> None
```

Open the audio card using GStreamer.

<a id="reachy_mini.media.webrtc_client_gstreamer.GstWebRTCClient.stop_recording"></a>

#### stop\_recording

```python
def stop_recording() -> None
```

Release the camera resource.

<a id="reachy_mini.media.webrtc_client_gstreamer.GstWebRTCClient.start_playing"></a>

#### start\_playing

```python
def start_playing() -> None
```

Open the audio output using GStreamer.

<a id="reachy_mini.media.webrtc_client_gstreamer.GstWebRTCClient.stop_playing"></a>

#### stop\_playing

```python
def stop_playing() -> None
```

Stop playing audio and release resources.

<a id="reachy_mini.media.webrtc_client_gstreamer.GstWebRTCClient.push_audio_sample"></a>

#### push\_audio\_sample

```python
def push_audio_sample(data: npt.NDArray[np.float32]) -> None
```

Push audio data to the output device.

<a id="reachy_mini.media.webrtc_client_gstreamer.GstWebRTCClient.play_sound"></a>

#### play\_sound

```python
def play_sound(sound_file: str) -> None
```

Play a sound file.

**Arguments**:

- `sound_file` _str_ - Path to the sound file to play.

<a id="reachy_mini.media.media_manager"></a>

# reachy\_mini.media.media\_manager

Media Manager.

Provides camera and audio access based on the selected backedn

<a id="reachy_mini.media.media_manager.MediaBackend"></a>

## MediaBackend Objects

```python
class MediaBackend(Enum)
```

Media backends.

<a id="reachy_mini.media.media_manager.MediaManager"></a>

## MediaManager Objects

```python
class MediaManager()
```

Abstract class for opening and managing audio devices.

<a id="reachy_mini.media.media_manager.MediaManager.__init__"></a>

#### \_\_init\_\_

```python
def __init__(backend: MediaBackend = MediaBackend.DEFAULT,
             log_level: str = "INFO",
             use_sim: bool = False,
             resolution: CameraResolution = CameraResolution.R1280x720,
             signalling_host: str = "localhost") -> None
```

Initialize the audio device.

<a id="reachy_mini.media.media_manager.MediaManager.__del__"></a>

#### \_\_del\_\_

```python
def __del__() -> None
```

Destructor to ensure resources are released.

<a id="reachy_mini.media.media_manager.MediaManager.get_frame"></a>

#### get\_frame

```python
def get_frame() -> Optional[npt.NDArray[np.uint8]]
```

Get a frame from the camera.

**Returns**:

- `Optional[npt.NDArray[np.uint8]]` - The captured BGR frame, or None if the camera is not available.

<a id="reachy_mini.media.media_manager.MediaManager.play_sound"></a>

#### play\_sound

```python
def play_sound(sound_file: str) -> None
```

Play a sound file.

**Arguments**:

- `sound_file` _str_ - Path to the sound file to play.

<a id="reachy_mini.media.media_manager.MediaManager.start_recording"></a>

#### start\_recording

```python
def start_recording() -> None
```

Start recording audio.

<a id="reachy_mini.media.media_manager.MediaManager.get_audio_sample"></a>

#### get\_audio\_sample

```python
def get_audio_sample() -> Optional[bytes | npt.NDArray[np.float32]]
```

Get an audio sample from the audio device.

**Returns**:

- `Optional[np.ndarray]` - The recorded audio sample, or None if no data is available.

<a id="reachy_mini.media.media_manager.MediaManager.get_audio_samplerate"></a>

#### get\_audio\_samplerate

```python
def get_audio_samplerate() -> int
```

Get the samplerate of the audio device.

**Returns**:

- `int` - The samplerate of the audio device.

<a id="reachy_mini.media.media_manager.MediaManager.stop_recording"></a>

#### stop\_recording

```python
def stop_recording() -> None
```

Stop recording audio.

<a id="reachy_mini.media.media_manager.MediaManager.start_playing"></a>

#### start\_playing

```python
def start_playing() -> None
```

Start playing audio.

<a id="reachy_mini.media.media_manager.MediaManager.push_audio_sample"></a>

#### push\_audio\_sample

```python
def push_audio_sample(data: npt.NDArray[np.float32]) -> None
```

Push audio data to the output device.

**Arguments**:

- `data` _npt.NDArray[np.float32]_ - The audio data to push to the output device (mono format).

<a id="reachy_mini.media.media_manager.MediaManager.stop_playing"></a>

#### stop\_playing

```python
def stop_playing() -> None
```

Stop playing audio.

<a id="reachy_mini.media.camera_utils"></a>

# reachy\_mini.media.camera\_utils

Camera utility for Reachy Mini.

<a id="reachy_mini.media.camera_utils.find_camera"></a>

#### find\_camera

```python
def find_camera(vid: int = 0x0C45,
                pid: int = 0x636D,
                apiPreference: int = cv2.CAP_ANY) -> cv2.VideoCapture | None
```

Find and return a camera with the specified VID and PID.

**Arguments**:

- `vid` _int_ - Vendor ID of the camera. Default is 0x0C45 (Arducam).
- `pid` _int_ - Product ID of the camera. Default is 0x636D (Arducam).
- `apiPreference` _int_ - Preferred API backend for the camera. Default is cv2.CAP_ANY.
  

**Returns**:

  cv2.VideoCapture | None: A VideoCapture object if the camera is found and opened successfully, otherwise None.

<a id="reachy_mini.media.camera_constants"></a>

# reachy\_mini.media.camera\_constants

Camera constants for Reachy Mini.

<a id="reachy_mini.media.camera_constants.CameraResolution"></a>

## CameraResolution Objects

```python
class CameraResolution(Enum)
```

Camera resolutions. Arducam_12MP.

<a id="reachy_mini.media.camera_constants.RPICameraResolution"></a>

## RPICameraResolution Objects

```python
class RPICameraResolution(Enum)
```

Camera resolutions. Raspberry Pi Camera.

Camera supports higher resolutions but the h264 encoder won't follow.

<a id="reachy_mini.media.camera_opencv"></a>

# reachy\_mini.media.camera\_opencv

OpenCv camera backend.

This module provides an implementation of the CameraBase class using OpenCV.

<a id="reachy_mini.media.camera_opencv.OpenCVCamera"></a>

## OpenCVCamera Objects

```python
class OpenCVCamera(CameraBase)
```

Camera implementation using OpenCV.

<a id="reachy_mini.media.camera_opencv.OpenCVCamera.__init__"></a>

#### \_\_init\_\_

```python
def __init__(
        log_level: str = "INFO",
        resolution: CameraResolution = CameraResolution.R1280x720) -> None
```

Initialize the OpenCV camera.

<a id="reachy_mini.media.camera_opencv.OpenCVCamera.open"></a>

#### open

```python
def open(udp_camera: Optional[str] = None) -> None
```

Open the camera using OpenCV VideoCapture.

<a id="reachy_mini.media.camera_opencv.OpenCVCamera.read"></a>

#### read

```python
def read() -> Optional[npt.NDArray[np.uint8]]
```

Read a frame from the camera. Returns the frame or None if error.

<a id="reachy_mini.media.camera_opencv.OpenCVCamera.close"></a>

#### close

```python
def close() -> None
```

Release the camera resource.

<a id="reachy_mini.media.webrtc_daemon"></a>

# reachy\_mini.media.webrtc\_daemon

WebRTC daemon.

Starts a gstreamer webrtc pipeline to stream video and audio.

<a id="reachy_mini.media.webrtc_daemon.GstWebRTC"></a>

## GstWebRTC Objects

```python
class GstWebRTC()
```

WebRTC pipeline using GStreamer.

<a id="reachy_mini.media.webrtc_daemon.GstWebRTC.__init__"></a>

#### \_\_init\_\_

```python
def __init__(
        log_level: str = "INFO",
        resolution: RPICameraResolution = RPICameraResolution.R1280x720
) -> None
```

Initialize the GStreamer WebRTC pipeline.

<a id="reachy_mini.media.webrtc_daemon.GstWebRTC.__del__"></a>

#### \_\_del\_\_

```python
def __del__() -> None
```

Destructor to ensure gstreamer resources are released.

<a id="reachy_mini.media.webrtc_daemon.GstWebRTC.resolution"></a>

#### resolution

```python
@property
def resolution() -> tuple[int, int]
```

Get the current camera resolution as a tuple (width, height).

<a id="reachy_mini.media.webrtc_daemon.GstWebRTC.framerate"></a>

#### framerate

```python
@property
def framerate() -> int
```

Get the current camera framerate.

<a id="reachy_mini.media.webrtc_daemon.GstWebRTC.start"></a>

#### start

```python
def start() -> None
```

Start the WebRTC pipeline.

<a id="reachy_mini.media.webrtc_daemon.GstWebRTC.pause"></a>

#### pause

```python
def pause() -> None
```

Pause the WebRTC pipeline.

<a id="reachy_mini.media.webrtc_daemon.GstWebRTC.stop"></a>

#### stop

```python
def stop() -> None
```

Stop the WebRTC pipeline.

<a id="reachy_mini.media.audio_base"></a>

# reachy\_mini.media.audio\_base

Base classes for audio implementations.

The audio implementations support various backends and provide a unified
interface for audio input/output.

<a id="reachy_mini.media.audio_base.AudioBase"></a>

## AudioBase Objects

```python
class AudioBase(ABC)
```

Abstract class for opening and managing audio devices.

<a id="reachy_mini.media.audio_base.AudioBase.SAMPLE_RATE"></a>

#### SAMPLE\_RATE

respeaker samplerate

<a id="reachy_mini.media.audio_base.AudioBase.__init__"></a>

#### \_\_init\_\_

```python
def __init__(log_level: str = "INFO") -> None
```

Initialize the audio device.

<a id="reachy_mini.media.audio_base.AudioBase.__del__"></a>

#### \_\_del\_\_

```python
def __del__() -> None
```

Destructor to ensure resources are released.

<a id="reachy_mini.media.audio_base.AudioBase.start_recording"></a>

#### start\_recording

```python
@abstractmethod
def start_recording() -> None
```

Start recording audio.

<a id="reachy_mini.media.audio_base.AudioBase.get_audio_sample"></a>

#### get\_audio\_sample

```python
@abstractmethod
def get_audio_sample() -> Optional[npt.NDArray[np.float32]]
```

Read audio data from the device. Returns the data or None if error.

<a id="reachy_mini.media.audio_base.AudioBase.stop_recording"></a>

#### stop\_recording

```python
@abstractmethod
def stop_recording() -> None
```

Close the audio device and release resources.

<a id="reachy_mini.media.audio_base.AudioBase.start_playing"></a>

#### start\_playing

```python
@abstractmethod
def start_playing() -> None
```

Start playing audio.

<a id="reachy_mini.media.audio_base.AudioBase.push_audio_sample"></a>

#### push\_audio\_sample

```python
@abstractmethod
def push_audio_sample(data: npt.NDArray[np.float32]) -> None
```

Push audio data to the output device.

<a id="reachy_mini.media.audio_base.AudioBase.stop_playing"></a>

#### stop\_playing

```python
@abstractmethod
def stop_playing() -> None
```

Stop playing audio and release resources.

<a id="reachy_mini.media.audio_base.AudioBase.play_sound"></a>

#### play\_sound

```python
@abstractmethod
def play_sound(sound_file: str) -> None
```

Play a sound file.

**Arguments**:

- `sound_file` _str_ - Path to the sound file to play.

<a id="reachy_mini.media.audio_base.AudioBase.get_DoA"></a>

#### get\_DoA

```python
def get_DoA() -> tuple[float, bool] | None
```

Get the Direction of Arrival (DoA) value from the ReSpeaker device.

The spatial angle is given in radians:
0 radians is left, π/2 radians is front/back, π radians is right.

Note: The microphone array requires firmware version 2.1.0 or higher to support this feature.
The firmware is located in src/reachy_mini/assets/firmware/*.bin.
Refer to https://wiki.seeedstudio.com/respeaker_xvf3800_introduction/`update`-firmware for the upgrade process.

**Returns**:

- `tuple` - A tuple containing the DoA value as a float (radians) and the speech detection as a bool, or None if the device is not found.

<a id="reachy_mini.media.audio_gstreamer"></a>

# reachy\_mini.media.audio\_gstreamer

GStreamer camera backend.

This module provides an implementation of the CameraBase class using GStreamer.
By default the module directly returns JPEG images as output by the camera.

<a id="reachy_mini.media.audio_gstreamer.GStreamerAudio"></a>

## GStreamerAudio Objects

```python
class GStreamerAudio(AudioBase)
```

Audio implementation using GStreamer.

<a id="reachy_mini.media.audio_gstreamer.GStreamerAudio.__init__"></a>

#### \_\_init\_\_

```python
def __init__(log_level: str = "INFO") -> None
```

Initialize the GStreamer audio.

<a id="reachy_mini.media.audio_gstreamer.GStreamerAudio.__del__"></a>

#### \_\_del\_\_

```python
def __del__() -> None
```

Destructor to ensure gstreamer resources are released.

<a id="reachy_mini.media.audio_gstreamer.GStreamerAudio.start_recording"></a>

#### start\_recording

```python
def start_recording() -> None
```

Open the audio card using GStreamer.

<a id="reachy_mini.media.audio_gstreamer.GStreamerAudio.get_audio_sample"></a>

#### get\_audio\_sample

```python
def get_audio_sample() -> Optional[npt.NDArray[np.float32]]
```

Read a sample from the audio card. Returns the sample or None if error.

**Returns**:

- `Optional[npt.NDArray[np.float32]]` - The captured sample in raw format, or None if error.

<a id="reachy_mini.media.audio_gstreamer.GStreamerAudio.stop_recording"></a>

#### stop\_recording

```python
def stop_recording() -> None
```

Release the camera resource.

<a id="reachy_mini.media.audio_gstreamer.GStreamerAudio.start_playing"></a>

#### start\_playing

```python
def start_playing() -> None
```

Open the audio output using GStreamer.

<a id="reachy_mini.media.audio_gstreamer.GStreamerAudio.stop_playing"></a>

#### stop\_playing

```python
def stop_playing() -> None
```

Stop playing audio and release resources.

<a id="reachy_mini.media.audio_gstreamer.GStreamerAudio.push_audio_sample"></a>

#### push\_audio\_sample

```python
def push_audio_sample(data: npt.NDArray[np.float32]) -> None
```

Push audio data to the output device.

<a id="reachy_mini.media.audio_gstreamer.GStreamerAudio.play_sound"></a>

#### play\_sound

```python
def play_sound(sound_file: str) -> None
```

Play a sound file.

**Arguments**:

- `sound_file` _str_ - Path to the sound file to play.

<a id="reachy_mini.media.audio_sounddevice"></a>

# reachy\_mini.media.audio\_sounddevice

Audio implementation using sounddevice backend.

<a id="reachy_mini.media.audio_sounddevice.SoundDeviceAudio"></a>

## SoundDeviceAudio Objects

```python
class SoundDeviceAudio(AudioBase)
```

Audio device implementation using sounddevice.

<a id="reachy_mini.media.audio_sounddevice.SoundDeviceAudio.__init__"></a>

#### \_\_init\_\_

```python
def __init__(frames_per_buffer: int = 256, log_level: str = "INFO") -> None
```

Initialize the SoundDevice audio device.

<a id="reachy_mini.media.audio_sounddevice.SoundDeviceAudio.start_recording"></a>

#### start\_recording

```python
def start_recording() -> None
```

Open the audio input stream, using ReSpeaker card if available.

<a id="reachy_mini.media.audio_sounddevice.SoundDeviceAudio.get_audio_sample"></a>

#### get\_audio\_sample

```python
def get_audio_sample() -> Optional[npt.NDArray[np.float32]]
```

Read audio data from the buffer. Returns numpy array or None if empty.

<a id="reachy_mini.media.audio_sounddevice.SoundDeviceAudio.stop_recording"></a>

#### stop\_recording

```python
def stop_recording() -> None
```

Close the audio stream and release resources.

<a id="reachy_mini.media.audio_sounddevice.SoundDeviceAudio.push_audio_sample"></a>

#### push\_audio\_sample

```python
def push_audio_sample(data: npt.NDArray[np.float32]) -> None
```

Push audio data to the output device.

<a id="reachy_mini.media.audio_sounddevice.SoundDeviceAudio.start_playing"></a>

#### start\_playing

```python
def start_playing() -> None
```

Open the audio output stream.

<a id="reachy_mini.media.audio_sounddevice.SoundDeviceAudio.stop_playing"></a>

#### stop\_playing

```python
def stop_playing() -> None
```

Close the audio output stream.

<a id="reachy_mini.media.audio_sounddevice.SoundDeviceAudio.play_sound"></a>

#### play\_sound

```python
def play_sound(sound_file: str, autoclean: bool = False) -> None
```

Play a sound file from the assets directory or a given path using sounddevice and soundfile.

<a id="reachy_mini.media.audio_sounddevice.SoundDeviceAudio.get_output_device_id"></a>

#### get\_output\_device\_id

```python
def get_output_device_id(name_contains: str) -> int
```

Return the output device id whose name contains the given string (case-insensitive).

If not found, return the default output device id.

<a id="reachy_mini.media.audio_sounddevice.SoundDeviceAudio.get_input_device_id"></a>

#### get\_input\_device\_id

```python
def get_input_device_id(name_contains: str) -> int
```

Return the input device id whose name contains the given string (case-insensitive).

If not found, return the default input device id.

<a id="reachy_mini.media.camera_base"></a>

# reachy\_mini.media.camera\_base

Base classes for camera implementations.

The camera implementations support various backends and provide a unified
interface for capturing images.

<a id="reachy_mini.media.camera_base.CameraBase"></a>

## CameraBase Objects

```python
class CameraBase(ABC)
```

Abstract class for opening and managing a camera.

<a id="reachy_mini.media.camera_base.CameraBase.__init__"></a>

#### \_\_init\_\_

```python
def __init__(
        log_level: str = "INFO",
        resolution: CameraResolution = CameraResolution.R1280x720) -> None
```

Initialize the camera.

<a id="reachy_mini.media.camera_base.CameraBase.resolution"></a>

#### resolution

```python
@property
def resolution() -> tuple[int, int]
```

Get the current camera resolution as a tuple (width, height).

<a id="reachy_mini.media.camera_base.CameraBase.framerate"></a>

#### framerate

```python
@property
def framerate() -> int
```

Get the current camera frames per second.

<a id="reachy_mini.media.camera_base.CameraBase.open"></a>

#### open

```python
@abstractmethod
def open() -> None
```

Open the camera.

<a id="reachy_mini.media.camera_base.CameraBase.read"></a>

#### read

```python
@abstractmethod
def read() -> Optional[npt.NDArray[np.uint8]]
```

Read an image from the camera. Returns the image or None if error.

<a id="reachy_mini.media.camera_base.CameraBase.close"></a>

#### close

```python
@abstractmethod
def close() -> None
```

Close the camera and release resources.

<a id="reachy_mini.media.audio_utils"></a>

# reachy\_mini.media.audio\_utils

Utility functions for audio handling, specifically for detecting the ReSpeaker sound card.

<a id="reachy_mini.media.audio_utils.get_respeaker_card_number"></a>

#### get\_respeaker\_card\_number

```python
def get_respeaker_card_number() -> int
```

Return the card number of the ReSpeaker sound card, or 0 if not found.

<a id="reachy_mini.media"></a>

# reachy\_mini.media

Media module.

<a id="reachy_mini.io.abstract"></a>

# reachy\_mini.io.abstract

Base classes for server and client implementations.

These abstract classes define the interface for server and client components
in the Reachy Mini project. They provide methods for starting and stopping
the server, handling commands, and managing client connections.

<a id="reachy_mini.io.abstract.AbstractServer"></a>

## AbstractServer Objects

```python
class AbstractServer(ABC)
```

Base class for server implementations.

<a id="reachy_mini.io.abstract.AbstractServer.start"></a>

#### start

```python
@abstractmethod
def start() -> None
```

Start the server.

<a id="reachy_mini.io.abstract.AbstractServer.stop"></a>

#### stop

```python
@abstractmethod
def stop() -> None
```

Stop the server.

<a id="reachy_mini.io.abstract.AbstractServer.command_received_event"></a>

#### command\_received\_event

```python
@abstractmethod
def command_received_event() -> Event
```

Wait for a new command and return it.

<a id="reachy_mini.io.abstract.AbstractClient"></a>

## AbstractClient Objects

```python
class AbstractClient(ABC)
```

Base class for client implementations.

<a id="reachy_mini.io.abstract.AbstractClient.wait_for_connection"></a>

#### wait\_for\_connection

```python
@abstractmethod
def wait_for_connection() -> None
```

Wait for the client to connect to the server.

<a id="reachy_mini.io.abstract.AbstractClient.is_connected"></a>

#### is\_connected

```python
@abstractmethod
def is_connected() -> bool
```

Check if the client is connected to the server.

<a id="reachy_mini.io.abstract.AbstractClient.disconnect"></a>

#### disconnect

```python
@abstractmethod
def disconnect() -> None
```

Disconnect the client from the server.

<a id="reachy_mini.io.abstract.AbstractClient.send_command"></a>

#### send\_command

```python
@abstractmethod
def send_command(command: str) -> None
```

Send a command to the server.

<a id="reachy_mini.io.abstract.AbstractClient.get_current_joints"></a>

#### get\_current\_joints

```python
@abstractmethod
def get_current_joints() -> tuple[list[float], list[float]]
```

Get the current joint positions.

<a id="reachy_mini.io.abstract.AbstractClient.send_task_request"></a>

#### send\_task\_request

```python
@abstractmethod
def send_task_request(task_req: AnyTaskRequest) -> UUID
```

Send a task request to the server and return a unique task identifier.

<a id="reachy_mini.io.abstract.AbstractClient.wait_for_task_completion"></a>

#### wait\_for\_task\_completion

```python
@abstractmethod
def wait_for_task_completion(task_uid: UUID, timeout: float = 5.0) -> None
```

Wait for the specified task to complete.

<a id="reachy_mini.io.zenoh_client"></a>

# reachy\_mini.io.zenoh\_client

Zenoh client for Reachy Mini.

This module implements a Zenoh client that allows communication with the Reachy Mini
robot. It subscribes to joint positions updates and allows sending commands to the robot.

<a id="reachy_mini.io.zenoh_client.ZenohClient"></a>

## ZenohClient Objects

```python
class ZenohClient(AbstractClient)
```

Zenoh client for Reachy Mini.

<a id="reachy_mini.io.zenoh_client.ZenohClient.__init__"></a>

#### \_\_init\_\_

```python
def __init__(localhost_only: bool = True)
```

Initialize the Zenoh client.

<a id="reachy_mini.io.zenoh_client.ZenohClient.wait_for_connection"></a>

#### wait\_for\_connection

```python
def wait_for_connection(timeout: float = 5.0) -> None
```

Wait for the client to connect to the server.

**Arguments**:

- `timeout` _float_ - Maximum time to wait for the connection in seconds.
  

**Raises**:

- `TimeoutError` - If the connection is not established within the timeout period.

<a id="reachy_mini.io.zenoh_client.ZenohClient.check_alive"></a>

#### check\_alive

```python
def check_alive() -> None
```

Periodically check if the client is still connected to the server.

<a id="reachy_mini.io.zenoh_client.ZenohClient.is_connected"></a>

#### is\_connected

```python
def is_connected() -> bool
```

Check if the client is connected to the server.

<a id="reachy_mini.io.zenoh_client.ZenohClient.disconnect"></a>

#### disconnect

```python
def disconnect() -> None
```

Disconnect the client from the server.

<a id="reachy_mini.io.zenoh_client.ZenohClient.send_command"></a>

#### send\_command

```python
def send_command(command: str) -> None
```

Send a command to the server.

<a id="reachy_mini.io.zenoh_client.ZenohClient.get_current_joints"></a>

#### get\_current\_joints

```python
def get_current_joints() -> tuple[list[float], list[float]]
```

Get the current joint positions.

<a id="reachy_mini.io.zenoh_client.ZenohClient.wait_for_recorded_data"></a>

#### wait\_for\_recorded\_data

```python
def wait_for_recorded_data(timeout: float = 5.0) -> bool
```

Block until the daemon publishes the frames (or timeout).

<a id="reachy_mini.io.zenoh_client.ZenohClient.get_recorded_data"></a>

#### get\_recorded\_data

```python
def get_recorded_data(
    wait: bool = True,
    timeout: float = 5.0
) -> Optional[List[Dict[str, float | List[float] | List[List[float]]]]]
```

Return the cached recording, optionally blocking until it arrives.

Raises `TimeoutError` if nothing shows up in time.

<a id="reachy_mini.io.zenoh_client.ZenohClient.get_status"></a>

#### get\_status

```python
def get_status(wait: bool = True, timeout: float = 5.0) -> Dict[str, Any]
```

Get the last received status. Returns DaemonStatus as a dict.

<a id="reachy_mini.io.zenoh_client.ZenohClient.get_current_head_pose"></a>

#### get\_current\_head\_pose

```python
def get_current_head_pose() -> npt.NDArray[np.float64]
```

Get the current head pose.

<a id="reachy_mini.io.zenoh_client.ZenohClient.send_task_request"></a>

#### send\_task\_request

```python
def send_task_request(task_req: AnyTaskRequest) -> UUID
```

Send a task request to the server.

<a id="reachy_mini.io.zenoh_client.ZenohClient.wait_for_task_completion"></a>

#### wait\_for\_task\_completion

```python
def wait_for_task_completion(task_uid: UUID, timeout: float = 5.0) -> None
```

Wait for the specified task to complete.

<a id="reachy_mini.io.zenoh_client.TaskState"></a>

## TaskState Objects

```python
@dataclass
class TaskState()
```

Represents the state of a task.

<a id="reachy_mini.io"></a>

# reachy\_mini.io

IO module.

<a id="reachy_mini.io.zenoh_server"></a>

# reachy\_mini.io.zenoh\_server

Zenoh server for Reachy Mini.

This module implements a Zenoh server that allows communication with the Reachy Mini
robot. It handles commands for joint positions and torque settings, and publishes joint positions updates.

It uses the Zenoh protocol for efficient data exchange and can be configured to run
either on localhost only or to accept connections from other hosts.

<a id="reachy_mini.io.zenoh_server.ZenohServer"></a>

## ZenohServer Objects

```python
class ZenohServer(AbstractServer)
```

Zenoh server for Reachy Mini.

<a id="reachy_mini.io.zenoh_server.ZenohServer.__init__"></a>

#### \_\_init\_\_

```python
def __init__(backend: Backend, localhost_only: bool = True)
```

Initialize the Zenoh server.

<a id="reachy_mini.io.zenoh_server.ZenohServer.start"></a>

#### start

```python
def start() -> None
```

Start the Zenoh server.

<a id="reachy_mini.io.zenoh_server.ZenohServer.stop"></a>

#### stop

```python
def stop() -> None
```

Stop the Zenoh server.

<a id="reachy_mini.io.zenoh_server.ZenohServer.command_received_event"></a>

#### command\_received\_event

```python
def command_received_event() -> threading.Event
```

Wait for a new command and return it.

<a id="reachy_mini.io.protocol"></a>

# reachy\_mini.io.protocol

Protocol definitions for Reachy Mini client/server communication.

<a id="reachy_mini.io.protocol.GotoTaskRequest"></a>

## GotoTaskRequest Objects

```python
class GotoTaskRequest(BaseModel)
```

Class to represent a goto target task.

<a id="reachy_mini.io.protocol.GotoTaskRequest.head"></a>

#### head

4x4 flatten pose matrix

<a id="reachy_mini.io.protocol.GotoTaskRequest.antennas"></a>

#### antennas

[right_angle, left_angle] (in rads)

<a id="reachy_mini.io.protocol.PlayMoveTaskRequest"></a>

## PlayMoveTaskRequest Objects

```python
class PlayMoveTaskRequest(BaseModel)
```

Class to represent a play move task.

<a id="reachy_mini.io.protocol.TaskRequest"></a>

## TaskRequest Objects

```python
class TaskRequest(BaseModel)
```

Class to represent any task request.

<a id="reachy_mini.io.protocol.TaskProgress"></a>

## TaskProgress Objects

```python
class TaskProgress(BaseModel)
```

Class to represent task progress.

<a id="reachy_mini.daemon.utils"></a>

# reachy\_mini.daemon.utils

Utilities for managing the Reachy Mini daemon.

<a id="reachy_mini.daemon.utils.daemon_check"></a>

#### daemon\_check

```python
def daemon_check(spawn_daemon: bool, use_sim: bool) -> None
```

Check if the Reachy Mini daemon is running and spawn it if necessary.

<a id="reachy_mini.daemon.utils.find_serial_port"></a>

#### find\_serial\_port

```python
def find_serial_port(wireless_version: bool = False,
                     vid: str = "1a86",
                     pid: str = "55d3",
                     pi_uart: str = "/dev/ttyAMA3") -> list[str]
```

Find the serial port for Reachy Mini based on VID and PID or the Raspberry Pi UART for the wireless version.

**Arguments**:

- `wireless_version` _bool_ - Whether to look for the wireless version using the Raspberry Pi UART.
- `vid` _str_ - Vendor ID of the device. (eg. "1a86").
- `pid` _str_ - Product ID of the device. (eg. "55d3").
- `pi_uart` _str_ - Path to the Raspberry Pi UART device. (eg. "/dev/ttyAMA3").

<a id="reachy_mini.daemon.utils.get_ip_address"></a>

#### get\_ip\_address

```python
def get_ip_address(ifname: str = "wlan0") -> str | None
```

Get the IP address of a specific network interface (Linux Only).

<a id="reachy_mini.daemon.utils.convert_enum_to_dict"></a>

#### convert\_enum\_to\_dict

```python
def convert_enum_to_dict(data: List[Any]) -> dict[str, Any]
```

Convert a dataclass containing Enums to a dictionary with enum values.

<a id="reachy_mini.daemon.app.routers.move"></a>

# reachy\_mini.daemon.app.routers.move

Movement-related API routes.

This exposes:
- goto
- play (wake_up, goto_sleep)
- stop running moves
- set_target and streaming set_target

<a id="reachy_mini.daemon.app.routers.move.InterpolationMode"></a>

## InterpolationMode Objects

```python
class InterpolationMode(str, Enum)
```

Interpolation modes for movement.

<a id="reachy_mini.daemon.app.routers.move.GotoModelRequest"></a>

## GotoModelRequest Objects

```python
class GotoModelRequest(BaseModel)
```

Request model for the goto endpoint.

<a id="reachy_mini.daemon.app.routers.move.MoveUUID"></a>

## MoveUUID Objects

```python
class MoveUUID(BaseModel)
```

Model representing a unique identifier for a move task.

<a id="reachy_mini.daemon.app.routers.move.create_move_task"></a>

#### create\_move\_task

```python
def create_move_task(coro: Coroutine[Any, Any, None]) -> MoveUUID
```

Create a new move task using async task coroutine.

<a id="reachy_mini.daemon.app.routers.move.stop_move_task"></a>

#### stop\_move\_task

```python
async def stop_move_task(uuid: UUID) -> dict[str, str]
```

Stop a running move task by cancelling it.

<a id="reachy_mini.daemon.app.routers.move.get_running_moves"></a>

#### get\_running\_moves

```python
@router.get("/running")
async def get_running_moves() -> list[MoveUUID]
```

Get a list of currently running move tasks.

<a id="reachy_mini.daemon.app.routers.move.goto"></a>

#### goto

```python
@router.post("/goto")
async def goto(
    goto_req: GotoModelRequest, backend: Backend = Depends(get_backend)
) -> MoveUUID
```

Request a movement to a specific target.

<a id="reachy_mini.daemon.app.routers.move.play_wake_up"></a>

#### play\_wake\_up

```python
@router.post("/play/wake_up")
async def play_wake_up(backend: Backend = Depends(get_backend)) -> MoveUUID
```

Request the robot to wake up.

<a id="reachy_mini.daemon.app.routers.move.play_goto_sleep"></a>

#### play\_goto\_sleep

```python
@router.post("/play/goto_sleep")
async def play_goto_sleep(backend: Backend = Depends(get_backend)) -> MoveUUID
```

Request the robot to go to sleep.

<a id="reachy_mini.daemon.app.routers.move.list_recorded_move_dataset"></a>

#### list\_recorded\_move\_dataset

```python
@router.get("/recorded-move-datasets/list/{dataset_name:path}")
async def list_recorded_move_dataset(dataset_name: str) -> list[str]
```

List available recorded moves in a dataset.

<a id="reachy_mini.daemon.app.routers.move.play_recorded_move_dataset"></a>

#### play\_recorded\_move\_dataset

```python
@router.post("/play/recorded-move-dataset/{dataset_name:path}/{move_name}")
async def play_recorded_move_dataset(
    dataset_name: str, move_name: str,
    backend: Backend = Depends(get_backend)) -> MoveUUID
```

Request the robot to play a predefined recorded move from a dataset.

<a id="reachy_mini.daemon.app.routers.move.stop_move"></a>

#### stop\_move

```python
@router.post("/stop")
async def stop_move(uuid: MoveUUID) -> dict[str, str]
```

Stop a running move task.

<a id="reachy_mini.daemon.app.routers.move.ws_move_updates"></a>

#### ws\_move\_updates

```python
@router.websocket("/ws/updates")
async def ws_move_updates(websocket: WebSocket) -> None
```

WebSocket route to stream move updates.

<a id="reachy_mini.daemon.app.routers.move.set_target"></a>

#### set\_target

```python
@router.post("/set_target")
async def set_target(
    target: FullBodyTarget, backend: Backend = Depends(get_backend)
) -> dict[str, str]
```

POST route to set a single FullBodyTarget.

<a id="reachy_mini.daemon.app.routers.move.ws_set_target"></a>

#### ws\_set\_target

```python
@router.websocket("/ws/set_target")
async def ws_set_target(
    websocket: WebSocket, backend: Backend = Depends(ws_get_backend)) -> None
```

WebSocket route to stream FullBodyTarget set_target calls.

<a id="reachy_mini.daemon.app.routers.wifi_config"></a>

# reachy\_mini.daemon.app.routers.wifi\_config

WiFi Configuration Routers.

<a id="reachy_mini.daemon.app.routers.wifi_config.WifiMode"></a>

## WifiMode Objects

```python
class WifiMode(Enum)
```

WiFi possible modes.

<a id="reachy_mini.daemon.app.routers.wifi_config.WifiStatus"></a>

## WifiStatus Objects

```python
class WifiStatus(BaseModel)
```

WiFi status model.

<a id="reachy_mini.daemon.app.routers.wifi_config.get_current_wifi_mode"></a>

#### get\_current\_wifi\_mode

```python
def get_current_wifi_mode() -> WifiMode
```

Get the current WiFi mode.

<a id="reachy_mini.daemon.app.routers.wifi_config.get_wifi_status"></a>

#### get\_wifi\_status

```python
@router.get("/status")
def get_wifi_status() -> WifiStatus
```

Get the current WiFi status.

<a id="reachy_mini.daemon.app.routers.wifi_config.get_last_wifi_error"></a>

#### get\_last\_wifi\_error

```python
@router.get("/error")
def get_last_wifi_error() -> dict[str, str | None]
```

Get the last WiFi error.

<a id="reachy_mini.daemon.app.routers.wifi_config.reset_last_wifi_error"></a>

#### reset\_last\_wifi\_error

```python
@router.post("/reset_error")
def reset_last_wifi_error() -> dict[str, str]
```

Reset the last WiFi error.

<a id="reachy_mini.daemon.app.routers.wifi_config.setup_hotspot"></a>

#### setup\_hotspot

```python
@router.post("/setup_hotspot")
def setup_hotspot(ssid: str = HOTSPOT_SSID,
                  password: str = HOTSPOT_PASSWORD) -> None
```

Set up a WiFi hotspot. It will create a new hotspot using nmcli if one does not already exist.

<a id="reachy_mini.daemon.app.routers.wifi_config.connect_to_wifi_network"></a>

#### connect\_to\_wifi\_network

```python
@router.post("/connect")
def connect_to_wifi_network(ssid: str, password: str) -> None
```

Connect to a WiFi network. It will create a new connection using nmcli if the specified SSID is not already configured.

<a id="reachy_mini.daemon.app.routers.wifi_config.scan_wifi"></a>

#### scan\_wifi

```python
@router.post("/scan_and_list")
def scan_wifi() -> list[str]
```

Scan for available WiFi networks ordered by signal power.

<a id="reachy_mini.daemon.app.routers.wifi_config.scan_available_wifi"></a>

#### scan\_available\_wifi

```python
def scan_available_wifi() -> list[nmcli.data.device.DeviceWifi]
```

Scan for available WiFi networks.

<a id="reachy_mini.daemon.app.routers.wifi_config.get_wifi_connections"></a>

#### get\_wifi\_connections

```python
def get_wifi_connections() -> list[nmcli.data.connection.Connection]
```

Get the list of WiFi connection.

<a id="reachy_mini.daemon.app.routers.wifi_config.check_if_connection_exists"></a>

#### check\_if\_connection\_exists

```python
def check_if_connection_exists(name: str) -> bool
```

Check if a WiFi connection with the given SSID already exists.

<a id="reachy_mini.daemon.app.routers.wifi_config.check_if_connection_active"></a>

#### check\_if\_connection\_active

```python
def check_if_connection_active(name: str) -> bool
```

Check if a WiFi connection with the given SSID is currently active.

<a id="reachy_mini.daemon.app.routers.wifi_config.setup_wifi_connection"></a>

#### setup\_wifi\_connection

```python
def setup_wifi_connection(name: str,
                          ssid: str,
                          password: str,
                          is_hotspot: bool = False) -> None
```

Set up a WiFi connection using nmcli.

<a id="reachy_mini.daemon.app.routers.wifi_config.remove_connection"></a>

#### remove\_connection

```python
def remove_connection(name: str) -> None
```

Remove a WiFi connection using nmcli.

<a id="reachy_mini.daemon.app.routers.update"></a>

# reachy\_mini.daemon.app.routers.update

Update router for Reachy Mini Daemon API.

This module provides endpoints to check for updates, start updates, and monitor update status.

<a id="reachy_mini.daemon.app.routers.update.available"></a>

#### available

```python
@router.get("/available")
def available(pre_release: bool = False) -> dict[str, dict[str, bool]]
```

Check if an update is available for Reachy Mini Wireless.

<a id="reachy_mini.daemon.app.routers.update.start_update"></a>

#### start\_update

```python
@router.post("/start")
def start_update(pre_release: bool = False) -> dict[str, str]
```

Start the update process for Reachy Mini Wireless version.

<a id="reachy_mini.daemon.app.routers.update.get_update_info"></a>

#### get\_update\_info

```python
@router.get("/info")
def get_update_info(job_id: str) -> JobInfo
```

Get the info of an update job.

<a id="reachy_mini.daemon.app.routers.update.websocket_logs"></a>

#### websocket\_logs

```python
@router.websocket("/ws/logs")
async def websocket_logs(websocket: WebSocket, job_id: str) -> None
```

WebSocket endpoint to stream update logs in real time.

<a id="reachy_mini.daemon.app.routers.state"></a>

# reachy\_mini.daemon.app.routers.state

State-related API routes.

This exposes:
- basic get routes to retrieve most common fields
- full state and streaming state updates

<a id="reachy_mini.daemon.app.routers.state.get_head_pose"></a>

#### get\_head\_pose

```python
@router.get("/present_head_pose")
async def get_head_pose(
    use_pose_matrix: bool = False,
    backend: Backend = Depends(get_backend)
) -> AnyPose
```

Get the present head pose.

**Arguments**:

- `use_pose_matrix` _bool_ - Whether to use the pose matrix representation (4x4 flattened) or the translation + Euler angles representation (x, y, z, roll, pitch, yaw).
- `backend` _Backend_ - The backend instance.
  

**Returns**:

- `AnyPose` - The present head pose.

<a id="reachy_mini.daemon.app.routers.state.get_body_yaw"></a>

#### get\_body\_yaw

```python
@router.get("/present_body_yaw")
async def get_body_yaw(backend: Backend = Depends(get_backend)) -> float
```

Get the present body yaw (in radians).

<a id="reachy_mini.daemon.app.routers.state.get_antenna_joint_positions"></a>

#### get\_antenna\_joint\_positions

```python
@router.get("/present_antenna_joint_positions")
async def get_antenna_joint_positions(backend: Backend = Depends(
    get_backend)) -> tuple[float, float]
```

Get the present antenna joint positions (in radians) - (left, right).

<a id="reachy_mini.daemon.app.routers.state.get_full_state"></a>

#### get\_full\_state

```python
@router.get("/full")
async def get_full_state(
    with_control_mode: bool = True,
    with_head_pose: bool = True,
    with_target_head_pose: bool = False,
    with_head_joints: bool = False,
    with_target_head_joints: bool = False,
    with_body_yaw: bool = True,
    with_target_body_yaw: bool = False,
    with_antenna_positions: bool = True,
    with_target_antenna_positions: bool = False,
    with_passive_joints: bool = False,
    use_pose_matrix: bool = False,
    backend: Backend = Depends(get_backend)
) -> FullState
```

Get the full robot state, with optional fields.

<a id="reachy_mini.daemon.app.routers.state.ws_full_state"></a>

#### ws\_full\_state

```python
@router.websocket("/ws/full")
async def ws_full_state(
    websocket: WebSocket,
    frequency: float = 10.0,
    with_head_pose: bool = True,
    with_target_head_pose: bool = False,
    with_head_joints: bool = False,
    with_target_head_joints: bool = False,
    with_body_yaw: bool = True,
    with_target_body_yaw: bool = False,
    with_antenna_positions: bool = True,
    with_target_antenna_positions: bool = False,
    with_passive_joints: bool = False,
    use_pose_matrix: bool = False,
    backend: Backend = Depends(ws_get_backend)
) -> None
```

WebSocket endpoint to stream the full state of the robot.

<a id="reachy_mini.daemon.app.routers.apps"></a>

# reachy\_mini.daemon.app.routers.apps

Apps router for apps management.

<a id="reachy_mini.daemon.app.routers.apps.list_available_apps"></a>

#### list\_available\_apps

```python
@router.get("/list-available/{source_kind}")
async def list_available_apps(
    source_kind: SourceKind,
    app_manager: "AppManager" = Depends(get_app_manager)
) -> list[AppInfo]
```

List available apps (including not installed).

<a id="reachy_mini.daemon.app.routers.apps.list_all_available_apps"></a>

#### list\_all\_available\_apps

```python
@router.get("/list-available")
async def list_all_available_apps(app_manager: "AppManager" = Depends(
    get_app_manager)) -> list[AppInfo]
```

List all available apps (including not installed).

<a id="reachy_mini.daemon.app.routers.apps.install_app"></a>

#### install\_app

```python
@router.post("/install")
async def install_app(
    app_info: AppInfo, app_manager: "AppManager" = Depends(get_app_manager)
) -> dict[str, str]
```

Install a new app by its info (background, returns job_id).

<a id="reachy_mini.daemon.app.routers.apps.remove_app"></a>

#### remove\_app

```python
@router.post("/remove/{app_name}")
async def remove_app(
    app_name: str, app_manager: "AppManager" = Depends(get_app_manager)
) -> dict[str, str]
```

Remove an installed app by its name (background, returns job_id).

<a id="reachy_mini.daemon.app.routers.apps.job_status"></a>

#### job\_status

```python
@router.get("/job-status/{job_id}")
async def job_status(job_id: str) -> bg_job_register.JobInfo
```

Get status/logs for a job.

<a id="reachy_mini.daemon.app.routers.apps.ws_apps_manager"></a>

#### ws\_apps\_manager

```python
@router.websocket("/ws/apps-manager/{job_id}")
async def ws_apps_manager(websocket: WebSocket, job_id: str) -> None
```

WebSocket route to stream live job status/logs for a job, sending updates as soon as new logs are available.

<a id="reachy_mini.daemon.app.routers.apps.start_app"></a>

#### start\_app

```python
@router.post("/start-app/{app_name}")
async def start_app(
    app_name: str, app_manager: "AppManager" = Depends(get_app_manager)
) -> AppStatus
```

Start an app by its name.

<a id="reachy_mini.daemon.app.routers.apps.restart_app"></a>

#### restart\_app

```python
@router.post("/restart-current-app")
async def restart_app(app_manager: "AppManager" = Depends(
    get_app_manager)) -> AppStatus
```

Restart the currently running app.

<a id="reachy_mini.daemon.app.routers.apps.stop_app"></a>

#### stop\_app

```python
@router.post("/stop-current-app")
async def stop_app(app_manager: "AppManager" = Depends(
    get_app_manager)) -> None
```

Stop the currently running app.

<a id="reachy_mini.daemon.app.routers.apps.current_app_status"></a>

#### current\_app\_status

```python
@router.get("/current-app-status")
async def current_app_status(app_manager: "AppManager" = Depends(
    get_app_manager)) -> AppStatus | None
```

Get the status of the currently running app, if any.

<a id="reachy_mini.daemon.app.routers.kinematics"></a>

# reachy\_mini.daemon.app.routers.kinematics

Kinematics router for handling kinematics-related requests.

This module defines the API endpoints for interacting with the kinematics
subsystem of the robot. It provides endpoints for retrieving URDF representation,
and other kinematics-related information.

<a id="reachy_mini.daemon.app.routers.kinematics.get_kinematics_info"></a>

#### get\_kinematics\_info

```python
@router.get("/info")
async def get_kinematics_info(backend: Backend = Depends(get_backend)) -> dict[
        str, Any]
```

Get the current information of the kinematics.

<a id="reachy_mini.daemon.app.routers.kinematics.get_urdf"></a>

#### get\_urdf

```python
@router.get("/urdf")
async def get_urdf(backend: Backend = Depends(get_backend)) -> dict[str, str]
```

Get the URDF representation of the robot.

<a id="reachy_mini.daemon.app.routers.kinematics.get_stl_file"></a>

#### get\_stl\_file

```python
@router.get("/stl/{filename}")
async def get_stl_file(filename: Path) -> Response
```

Get the path to an STL asset file.

<a id="reachy_mini.daemon.app.routers.daemon"></a>

# reachy\_mini.daemon.app.routers.daemon

Daemon-related API routes.

<a id="reachy_mini.daemon.app.routers.daemon.start_daemon"></a>

#### start\_daemon

```python
@router.post("/start")
async def start_daemon(
    request: Request, wake_up: bool,
    daemon: Daemon = Depends(get_daemon)) -> dict[str, str]
```

Start the daemon.

<a id="reachy_mini.daemon.app.routers.daemon.stop_daemon"></a>

#### stop\_daemon

```python
@router.post("/stop")
async def stop_daemon(
    goto_sleep: bool, daemon: Daemon = Depends(get_daemon)) -> dict[str, str]
```

Stop the daemon, optionally putting the robot to sleep.

<a id="reachy_mini.daemon.app.routers.daemon.restart_daemon"></a>

#### restart\_daemon

```python
@router.post("/restart")
async def restart_daemon(
    request: Request, daemon: Daemon = Depends(get_daemon)) -> dict[str, str]
```

Restart the daemon.

<a id="reachy_mini.daemon.app.routers.daemon.get_daemon_status"></a>

#### get\_daemon\_status

```python
@router.get("/status")
async def get_daemon_status(daemon: Daemon = Depends(
    get_daemon)) -> DaemonStatus
```

Get the current status of the daemon.

<a id="reachy_mini.daemon.app.routers.motors"></a>

# reachy\_mini.daemon.app.routers.motors

Motors router.

Provides endpoints to get and set the motor control mode.

<a id="reachy_mini.daemon.app.routers.motors.MotorStatus"></a>

## MotorStatus Objects

```python
class MotorStatus(BaseModel)
```

Represents the status of the motors.

Exposes
- mode: The current motor control mode (enabled, disabled, gravity_compensation).

<a id="reachy_mini.daemon.app.routers.motors.get_motor_status"></a>

#### get\_motor\_status

```python
@router.get("/status")
async def get_motor_status(backend: Backend = Depends(
    get_backend)) -> MotorStatus
```

Get the current status of the motors.

<a id="reachy_mini.daemon.app.routers.motors.set_motor_mode"></a>

#### set\_motor\_mode

```python
@router.post("/set_mode/{mode}")
async def set_motor_mode(
    mode: MotorControlMode, backend: Backend = Depends(get_backend)
) -> dict[str, str]
```

Set the motor control mode.

<a id="reachy_mini.daemon.app.services.bluetooth.bluetooth_service"></a>

# reachy\_mini.daemon.app.services.bluetooth.bluetooth\_service

Bluetooth service for Reachy Mini using direct DBus API.

Includes a fixed NoInputNoOutput agent for automatic Just Works pairing.

<a id="reachy_mini.daemon.app.services.bluetooth.bluetooth_service.NoInputAgent"></a>

## NoInputAgent Objects

```python
class NoInputAgent(dbus.service.Object)
```

BLE Agent for Just Works pairing.

<a id="reachy_mini.daemon.app.services.bluetooth.bluetooth_service.NoInputAgent.Release"></a>

#### Release

```python
@dbus.service.method("org.bluez.Agent1", in_signature="", out_signature="")
def Release(*args)
```

Handle release of the agent.

<a id="reachy_mini.daemon.app.services.bluetooth.bluetooth_service.NoInputAgent.RequestPinCode"></a>

#### RequestPinCode

```python
@dbus.service.method("org.bluez.Agent1", in_signature="", out_signature="s")
def RequestPinCode(*args)
```

Automatically provide an empty pin code for Just Works pairing.

<a id="reachy_mini.daemon.app.services.bluetooth.bluetooth_service.NoInputAgent.RequestPasskey"></a>

#### RequestPasskey

```python
@dbus.service.method("org.bluez.Agent1", in_signature="", out_signature="u")
def RequestPasskey(*args)
```

Automatically provide a passkey of 0 for Just Works pairing.

<a id="reachy_mini.daemon.app.services.bluetooth.bluetooth_service.NoInputAgent.RequestConfirmation"></a>

#### RequestConfirmation

```python
@dbus.service.method("org.bluez.Agent1", in_signature="", out_signature="")
def RequestConfirmation(*args)
```

Automatically confirm the pairing request.

<a id="reachy_mini.daemon.app.services.bluetooth.bluetooth_service.NoInputAgent.DisplayPinCode"></a>

#### DisplayPinCode

```python
@dbus.service.method("org.bluez.Agent1", in_signature="", out_signature="")
def DisplayPinCode(*args)
```

Handle displaying the pin code (not used in Just Works).

<a id="reachy_mini.daemon.app.services.bluetooth.bluetooth_service.NoInputAgent.DisplayPasskey"></a>

#### DisplayPasskey

```python
@dbus.service.method("org.bluez.Agent1", in_signature="", out_signature="")
def DisplayPasskey(*args)
```

Handle displaying the passkey (not used in Just Works).

<a id="reachy_mini.daemon.app.services.bluetooth.bluetooth_service.NoInputAgent.AuthorizeService"></a>

#### AuthorizeService

```python
@dbus.service.method("org.bluez.Agent1", in_signature="", out_signature="")
def AuthorizeService(*args)
```

Handle service authorization requests.

<a id="reachy_mini.daemon.app.services.bluetooth.bluetooth_service.NoInputAgent.Cancel"></a>

#### Cancel

```python
@dbus.service.method("org.bluez.Agent1", in_signature="", out_signature="")
def Cancel(*args)
```

Handle cancellation of the agent request.

<a id="reachy_mini.daemon.app.services.bluetooth.bluetooth_service.Advertisement"></a>

## Advertisement Objects

```python
class Advertisement(dbus.service.Object)
```

BLE Advertisement.

<a id="reachy_mini.daemon.app.services.bluetooth.bluetooth_service.Advertisement.__init__"></a>

#### \_\_init\_\_

```python
def __init__(bus, index, advertising_type, local_name)
```

Initialize the Advertisement.

<a id="reachy_mini.daemon.app.services.bluetooth.bluetooth_service.Advertisement.get_properties"></a>

#### get\_properties

```python
def get_properties()
```

Return the properties of the advertisement.

<a id="reachy_mini.daemon.app.services.bluetooth.bluetooth_service.Advertisement.get_path"></a>

#### get\_path

```python
def get_path()
```

Return the object path.

<a id="reachy_mini.daemon.app.services.bluetooth.bluetooth_service.Advertisement.GetAll"></a>

#### GetAll

```python
@dbus.service.method(DBUS_PROP_IFACE, in_signature="s", out_signature="a{sv}")
def GetAll(interface)
```

Return all properties of the advertisement.

<a id="reachy_mini.daemon.app.services.bluetooth.bluetooth_service.Advertisement.Release"></a>

#### Release

```python
@dbus.service.method(LE_ADVERTISEMENT_IFACE, in_signature="", out_signature="")
def Release()
```

Handle release of the advertisement.

<a id="reachy_mini.daemon.app.services.bluetooth.bluetooth_service.Characteristic"></a>

## Characteristic Objects

```python
class Characteristic(dbus.service.Object)
```

GATT Characteristic.

<a id="reachy_mini.daemon.app.services.bluetooth.bluetooth_service.Characteristic.__init__"></a>

#### \_\_init\_\_

```python
def __init__(bus, index, uuid, flags, service)
```

Initialize the Characteristic.

<a id="reachy_mini.daemon.app.services.bluetooth.bluetooth_service.Characteristic.get_properties"></a>

#### get\_properties

```python
def get_properties()
```

Return the properties of the characteristic.

<a id="reachy_mini.daemon.app.services.bluetooth.bluetooth_service.Characteristic.get_path"></a>

#### get\_path

```python
def get_path()
```

Return the object path.

<a id="reachy_mini.daemon.app.services.bluetooth.bluetooth_service.Characteristic.GetAll"></a>

#### GetAll

```python
@dbus.service.method(DBUS_PROP_IFACE, in_signature="s", out_signature="a{sv}")
def GetAll(interface)
```

Return all properties of the characteristic.

<a id="reachy_mini.daemon.app.services.bluetooth.bluetooth_service.Characteristic.ReadValue"></a>

#### ReadValue

```python
@dbus.service.method(GATT_CHRC_IFACE, in_signature="a{sv}", out_signature="ay")
def ReadValue(options)
```

Handle read from the characteristic.

<a id="reachy_mini.daemon.app.services.bluetooth.bluetooth_service.Characteristic.WriteValue"></a>

#### WriteValue

```python
@dbus.service.method(GATT_CHRC_IFACE, in_signature="aya{sv}")
def WriteValue(value, options)
```

Handle write to the characteristic.

<a id="reachy_mini.daemon.app.services.bluetooth.bluetooth_service.CommandCharacteristic"></a>

## CommandCharacteristic Objects

```python
class CommandCharacteristic(Characteristic)
```

Command Characteristic.

<a id="reachy_mini.daemon.app.services.bluetooth.bluetooth_service.CommandCharacteristic.__init__"></a>

#### \_\_init\_\_

```python
def __init__(bus, index, service, command_handler: Callable[[bytes], str])
```

Initialize the Command Characteristic.

<a id="reachy_mini.daemon.app.services.bluetooth.bluetooth_service.CommandCharacteristic.WriteValue"></a>

#### WriteValue

```python
def WriteValue(value, options)
```

Handle write to the Command Characteristic.

<a id="reachy_mini.daemon.app.services.bluetooth.bluetooth_service.ResponseCharacteristic"></a>

## ResponseCharacteristic Objects

```python
class ResponseCharacteristic(Characteristic)
```

<a id="reachy_mini.daemon.app.services.bluetooth.bluetooth_service.ResponseCharacteristic.__init__"></a>

#### \_\_init\_\_

```python
def __init__(bus, index, service)
```

Initialize the Response Characteristic.

<a id="reachy_mini.daemon.app.services.bluetooth.bluetooth_service.Service"></a>

## Service Objects

```python
class Service(dbus.service.Object)
```

GATT Service.

<a id="reachy_mini.daemon.app.services.bluetooth.bluetooth_service.Service.__init__"></a>

#### \_\_init\_\_

```python
def __init__(bus, index, uuid, primary, command_handler: Callable[[bytes],
                                                                  str])
```

Initialize the GATT Service.

<a id="reachy_mini.daemon.app.services.bluetooth.bluetooth_service.Service.get_properties"></a>

#### get\_properties

```python
def get_properties()
```

Return the properties of the service.

<a id="reachy_mini.daemon.app.services.bluetooth.bluetooth_service.Service.get_path"></a>

#### get\_path

```python
def get_path()
```

Return the object path.

<a id="reachy_mini.daemon.app.services.bluetooth.bluetooth_service.Service.add_characteristic"></a>

#### add\_characteristic

```python
def add_characteristic(ch)
```

Add a characteristic to the service.

<a id="reachy_mini.daemon.app.services.bluetooth.bluetooth_service.Service.GetAll"></a>

#### GetAll

```python
@dbus.service.method(DBUS_PROP_IFACE, in_signature="s", out_signature="a{sv}")
def GetAll(interface)
```

Return all properties of the service.

<a id="reachy_mini.daemon.app.services.bluetooth.bluetooth_service.Application"></a>

## Application Objects

```python
class Application(dbus.service.Object)
```

GATT Application.

<a id="reachy_mini.daemon.app.services.bluetooth.bluetooth_service.Application.__init__"></a>

#### \_\_init\_\_

```python
def __init__(bus, command_handler: Callable[[bytes], str])
```

Initialize the GATT Application.

<a id="reachy_mini.daemon.app.services.bluetooth.bluetooth_service.Application.get_path"></a>

#### get\_path

```python
def get_path()
```

Return the object path.

<a id="reachy_mini.daemon.app.services.bluetooth.bluetooth_service.Application.GetManagedObjects"></a>

#### GetManagedObjects

```python
@dbus.service.method(DBUS_OM_IFACE, out_signature="a{oa{sa{sv}}}")
def GetManagedObjects()
```

Return a dictionary of all managed objects.

<a id="reachy_mini.daemon.app.services.bluetooth.bluetooth_service.BluetoothCommandService"></a>

## BluetoothCommandService Objects

```python
class BluetoothCommandService()
```

Bluetooth Command Service.

<a id="reachy_mini.daemon.app.services.bluetooth.bluetooth_service.BluetoothCommandService.__init__"></a>

#### \_\_init\_\_

```python
def __init__(device_name="ReachyMini")
```

Initialize the Bluetooth Command Service.

<a id="reachy_mini.daemon.app.services.bluetooth.bluetooth_service.BluetoothCommandService.start"></a>

#### start

```python
def start()
```

Start the Bluetooth Command Service.

<a id="reachy_mini.daemon.app.services.bluetooth.bluetooth_service.BluetoothCommandService.run"></a>

#### run

```python
def run()
```

Run the Bluetooth Command Service.

<a id="reachy_mini.daemon.app.services.bluetooth.bluetooth_service.main"></a>

#### main

```python
def main()
```

Run the Bluetooth Command Service.

<a id="reachy_mini.daemon.app.dependencies"></a>

# reachy\_mini.daemon.app.dependencies

FastAPI common request dependencies.

<a id="reachy_mini.daemon.app.dependencies.get_daemon"></a>

#### get\_daemon

```python
def get_daemon(request: Request) -> Daemon
```

Get the daemon as request dependency.

<a id="reachy_mini.daemon.app.dependencies.get_backend"></a>

#### get\_backend

```python
def get_backend(request: Request) -> Backend
```

Get the backend as request dependency.

<a id="reachy_mini.daemon.app.dependencies.get_app_manager"></a>

#### get\_app\_manager

```python
def get_app_manager(request: Request) -> "AppManager"
```

Get the app manager as request dependency.

<a id="reachy_mini.daemon.app.dependencies.ws_get_backend"></a>

#### ws\_get\_backend

```python
def ws_get_backend(websocket: WebSocket) -> Backend
```

Get the backend as websocket dependency.

<a id="reachy_mini.daemon.app.main"></a>

# reachy\_mini.daemon.app.main

Daemon entry point for the Reachy Mini robot.

This script serves as the command-line interface (CLI) entry point for the Reachy Mini daemon.
It initializes the daemon with specified parameters such as simulation mode, serial port,
scene to load, and logging level. The daemon runs indefinitely, handling requests and
managing the robot's state.

<a id="reachy_mini.daemon.app.main.Args"></a>

## Args Objects

```python
@dataclass
class Args()
```

Arguments for configuring the Reachy Mini daemon.

<a id="reachy_mini.daemon.app.main.create_app"></a>

#### create\_app

```python
def create_app(args: Args) -> FastAPI
```

Create and configure the FastAPI application.

<a id="reachy_mini.daemon.app.main.run_app"></a>

#### run\_app

```python
def run_app(args: Args) -> None
```

Run the FastAPI app with Uvicorn.

<a id="reachy_mini.daemon.app.main.main"></a>

#### main

```python
def main() -> None
```

Run the FastAPI app with Uvicorn.

<a id="reachy_mini.daemon.app"></a>

# reachy\_mini.daemon.app

FastAPI app initialization.

<a id="reachy_mini.daemon.app.bg_job_register"></a>

# reachy\_mini.daemon.app.bg\_job\_register

Background jobs management for Reachy Mini Daemon.

<a id="reachy_mini.daemon.app.bg_job_register.JobStatus"></a>

## JobStatus Objects

```python
class JobStatus(Enum)
```

Enum for job status.

<a id="reachy_mini.daemon.app.bg_job_register.JobInfo"></a>

## JobInfo Objects

```python
class JobInfo(BaseModel)
```

Pydantic model for install job status.

<a id="reachy_mini.daemon.app.bg_job_register.JobHandler"></a>

## JobHandler Objects

```python
@dataclass
class JobHandler()
```

Handler for background jobs.

<a id="reachy_mini.daemon.app.bg_job_register.run_command"></a>

#### run\_command

```python
def run_command(command: str, coro_func: Callable[..., Awaitable[None]], *args:
                Any) -> str
```

Start a background job, with a custom logger and return its job_id.

<a id="reachy_mini.daemon.app.bg_job_register.get_info"></a>

#### get\_info

```python
def get_info(job_id: str) -> JobInfo
```

Get the info of a job by its ID.

<a id="reachy_mini.daemon.app.bg_job_register.ws_poll_info"></a>

#### ws\_poll\_info

```python
async def ws_poll_info(websocket: WebSocket, job_uuid: str) -> None
```

WebSocket endpoint to stream job logs in real time.

<a id="reachy_mini.daemon.app.models"></a>

# reachy\_mini.daemon.app.models

Common pydantic models definitions.

<a id="reachy_mini.daemon.app.models.Matrix4x4Pose"></a>

## Matrix4x4Pose Objects

```python
class Matrix4x4Pose(BaseModel)
```

Represent a 3D pose by its 4x4 transformation matrix (translation is expressed in meters).

<a id="reachy_mini.daemon.app.models.Matrix4x4Pose.from_pose_array"></a>

#### from\_pose\_array

```python
@classmethod
def from_pose_array(cls, arr: NDArray[np.float64]) -> "Matrix4x4Pose"
```

Create a Matrix4x4 pose representation from a 4x4 pose array.

<a id="reachy_mini.daemon.app.models.Matrix4x4Pose.to_pose_array"></a>

#### to\_pose\_array

```python
def to_pose_array() -> NDArray[np.float64]
```

Convert the Matrix4x4Pose to a 4x4 numpy array.

<a id="reachy_mini.daemon.app.models.XYZRPYPose"></a>

## XYZRPYPose Objects

```python
class XYZRPYPose(BaseModel)
```

Represent a 3D pose using position (x, y, z) in meters and orientation (roll, pitch, yaw) angles in radians.

<a id="reachy_mini.daemon.app.models.XYZRPYPose.from_pose_array"></a>

#### from\_pose\_array

```python
@classmethod
def from_pose_array(cls, arr: NDArray[np.float64]) -> "XYZRPYPose"
```

Create an XYZRPYPose representation from a 4x4 pose array.

<a id="reachy_mini.daemon.app.models.XYZRPYPose.to_pose_array"></a>

#### to\_pose\_array

```python
def to_pose_array() -> NDArray[np.float64]
```

Convert the XYZRPYPose to a 4x4 numpy array.

<a id="reachy_mini.daemon.app.models.as_any_pose"></a>

#### as\_any\_pose

```python
def as_any_pose(pose: NDArray[np.float64], use_matrix: bool) -> AnyPose
```

Convert a numpy array to an AnyPose representation.

<a id="reachy_mini.daemon.app.models.FullBodyTarget"></a>

## FullBodyTarget Objects

```python
class FullBodyTarget(BaseModel)
```

Represent the full body including the head pose and the joints for antennas.

<a id="reachy_mini.daemon.app.models.FullState"></a>

## FullState Objects

```python
class FullState(BaseModel)
```

Represent the full state of the robot including all joint positions and poses.

<a id="reachy_mini.daemon.backend.abstract"></a>

# reachy\_mini.daemon.backend.abstract

Base class for robot backends, simulated or real.

This module defines the `Backend` class, which serves as a base for implementing
different types of robot backends, whether they are simulated (like Mujoco) or real
(connected via serial port). The class provides methods for managing joint positions,
torque control, and other backend-specific functionalities.
It is designed to be extended by subclasses that implement the specific behavior for
each type of backend.

<a id="reachy_mini.daemon.backend.abstract.MotorControlMode"></a>

## MotorControlMode Objects

```python
class MotorControlMode(str, Enum)
```

Enum for motor control modes.

<a id="reachy_mini.daemon.backend.abstract.MotorControlMode.Enabled"></a>

#### Enabled

Torque ON and controlled in position

<a id="reachy_mini.daemon.backend.abstract.MotorControlMode.Disabled"></a>

#### Disabled

Torque OFF

<a id="reachy_mini.daemon.backend.abstract.MotorControlMode.GravityCompensation"></a>

#### GravityCompensation

Torque ON and controlled in current to compensate for gravity

<a id="reachy_mini.daemon.backend.abstract.Backend"></a>

## Backend Objects

```python
class Backend()
```

Base class for robot backends, simulated or real.

<a id="reachy_mini.daemon.backend.abstract.Backend.__init__"></a>

#### \_\_init\_\_

```python
def __init__(log_level: str = "INFO",
             check_collision: bool = False,
             kinematics_engine: str = "AnalyticalKinematics") -> None
```

Initialize the backend.

<a id="reachy_mini.daemon.backend.abstract.Backend.wrapped_run"></a>

#### wrapped\_run

```python
def wrapped_run() -> None
```

Run the backend in a try-except block to store errors.

<a id="reachy_mini.daemon.backend.abstract.Backend.run"></a>

#### run

```python
def run() -> None
```

Run the backend.

This method is a placeholder and should be overridden by subclasses.

<a id="reachy_mini.daemon.backend.abstract.Backend.close"></a>

#### close

```python
def close() -> None
```

Close the backend.

This method is a placeholder and should be overridden by subclasses.

<a id="reachy_mini.daemon.backend.abstract.Backend.get_status"></a>

#### get\_status

```python
def get_status() -> "RobotBackendStatus | MujocoBackendStatus"
```

Return backend statistics.

This method is a placeholder and should be overridden by subclasses.

<a id="reachy_mini.daemon.backend.abstract.Backend.set_joint_positions_publisher"></a>

#### set\_joint\_positions\_publisher

```python
def set_joint_positions_publisher(publisher: zenoh.Publisher) -> None
```

Set the publisher for joint positions.

**Arguments**:

- `publisher` - A publisher object that will be used to publish joint positions.

<a id="reachy_mini.daemon.backend.abstract.Backend.set_pose_publisher"></a>

#### set\_pose\_publisher

```python
def set_pose_publisher(publisher: zenoh.Publisher) -> None
```

Set the publisher for head pose.

**Arguments**:

- `publisher` - A publisher object that will be used to publish head pose.

<a id="reachy_mini.daemon.backend.abstract.Backend.update_target_head_joints_from_ik"></a>

#### update\_target\_head\_joints\_from\_ik

```python
def update_target_head_joints_from_ik(pose: Annotated[NDArray[np.float64],
                                                      (4, 4)] | None = None,
                                      body_yaw: float | None = None) -> None
```

Update the target head joint positions from inverse kinematics.

**Arguments**:

- `pose` _np.ndarray_ - 4x4 pose matrix representing the head pose.
- `body_yaw` _float_ - The yaw angle of the body, used to adjust the head pose.

<a id="reachy_mini.daemon.backend.abstract.Backend.set_target_head_pose"></a>

#### set\_target\_head\_pose

```python
def set_target_head_pose(pose: Annotated[NDArray[np.float64], (4, 4)]) -> None
```

Set the target head pose for the robot.

**Arguments**:

- `pose` _np.ndarray_ - 4x4 pose matrix representing the head pose.

<a id="reachy_mini.daemon.backend.abstract.Backend.set_target_body_yaw"></a>

#### set\_target\_body\_yaw

```python
def set_target_body_yaw(body_yaw: float) -> None
```

Set the target body yaw for the robot.

Only used when doing a set_target() with a standalone body_yaw (no head pose).

**Arguments**:

- `body_yaw` _float_ - The yaw angle of the body

<a id="reachy_mini.daemon.backend.abstract.Backend.set_target_head_joint_positions"></a>

#### set\_target\_head\_joint\_positions

```python
def set_target_head_joint_positions(
        positions: Annotated[NDArray[np.float64], (7, )] | None) -> None
```

Set the head joint positions.

**Arguments**:

- `positions` _List[float]_ - A list of joint positions for the head.

<a id="reachy_mini.daemon.backend.abstract.Backend.set_target"></a>

#### set\_target

```python
def set_target(head: Annotated[NDArray[np.float64], (4, 4)] | None = None,
               antennas: Annotated[NDArray[np.float64], (2, )]
               | None = None,
               body_yaw: float | None = None) -> None
```

Set the target head pose and/or antenna positions and/or body_yaw.

<a id="reachy_mini.daemon.backend.abstract.Backend.set_target_antenna_joint_positions"></a>

#### set\_target\_antenna\_joint\_positions

```python
def set_target_antenna_joint_positions(
        positions: Annotated[NDArray[np.float64], (2, )]) -> None
```

Set the antenna joint positions.

**Arguments**:

- `positions` _List[float]_ - A list of joint positions for the antenna.

<a id="reachy_mini.daemon.backend.abstract.Backend.set_target_head_joint_current"></a>

#### set\_target\_head\_joint\_current

```python
def set_target_head_joint_current(
        current: Annotated[NDArray[np.float64], (7, )]) -> None
```

Set the head joint current.

**Arguments**:

  current (Annotated[NDArray[np.float64], (7,)]): A list of current values for the head motors.

<a id="reachy_mini.daemon.backend.abstract.Backend.play_move"></a>

#### play\_move

```python
async def play_move(move: Move,
                    play_frequency: float = 100.0,
                    initial_goto_duration: float = 0.0) -> None
```

Asynchronously play a Move.

**Arguments**:

- `move` _Move_ - The Move object to be played.
- `play_frequency` _float_ - The frequency at which to evaluate the move (in Hz).
- `initial_goto_duration` _float_ - Duration for an initial goto to the move's starting position. If 0.0, no initial goto is performed.

<a id="reachy_mini.daemon.backend.abstract.Backend.goto_target"></a>

#### goto\_target

```python
async def goto_target(
        head: Annotated[NDArray[np.float64], (4, 4)] | None = None,
        antennas: Annotated[NDArray[np.float64], (2, )]
    | None = None,
        duration: float = 0.5,
        method: InterpolationTechnique = InterpolationTechnique.MIN_JERK,
        body_yaw: float | None = 0.0) -> None
```

Asynchronously go to a target head pose and/or antennas position using task space interpolation, in "duration" seconds.

**Arguments**:

- `head` _np.ndarray | None_ - 4x4 pose matrix representing the target head pose.
- `antennas` _np.ndarray | list[float] | None_ - 1D array with two elements representing the angles of the antennas in radians.
- `duration` _float_ - Duration of the movement in seconds.
- `method` _str_ - Interpolation method to use ("linear", "minjerk", "ease", "cartoon"). Default is "minjerk".
- `body_yaw` _float | None_ - Body yaw angle in radians.
  

**Raises**:

- `ValueError` - If neither head nor antennas are provided, or if duration is not positive.

<a id="reachy_mini.daemon.backend.abstract.Backend.goto_joint_positions"></a>

#### goto\_joint\_positions

```python
async def goto_joint_positions(
        head_joint_positions: list[float]
    | None = None,
        antennas_joint_positions: list[float]
    | None = None,
        duration: float = 0.5,
        method: InterpolationTechnique = InterpolationTechnique.MIN_JERK
) -> None
```

Asynchronously go to a target head joint positions and/or antennas joint positions using joint space interpolation, in "duration" seconds.

Go to a target head joint positions and/or antennas joint positions using joint space interpolation, in "duration" seconds.

**Arguments**:

- `head_joint_positions` _Optional[List[float]]_ - List of head joint positions in radians (length 7).
- `antennas_joint_positions` _Optional[List[float]]_ - List of antennas joint positions in radians (length 2).
- `duration` _float_ - Duration of the movement in seconds. Default is 0.5 seconds.
- `method` _str_ - Interpolation method to use ("linear", "minjerk", "ease", "cartoon"). Default is "minjerk".
  

**Raises**:

- `ValueError` - If neither head_joint_positions nor antennas_joint_positions are provided, or if duration is not positive.

<a id="reachy_mini.daemon.backend.abstract.Backend.set_recording_publisher"></a>

#### set\_recording\_publisher

```python
def set_recording_publisher(publisher: zenoh.Publisher) -> None
```

Set the publisher for recording data.

**Arguments**:

- `publisher` - A publisher object that will be used to publish recorded data.

<a id="reachy_mini.daemon.backend.abstract.Backend.append_record"></a>

#### append\_record

```python
def append_record(record: dict[str, Any]) -> None
```

Append a record to the recorded data.

**Arguments**:

- `record` _dict_ - A dictionary containing the record data to be appended.

<a id="reachy_mini.daemon.backend.abstract.Backend.start_recording"></a>

#### start\_recording

```python
def start_recording() -> None
```

Start recording data.

<a id="reachy_mini.daemon.backend.abstract.Backend.stop_recording"></a>

#### stop\_recording

```python
def stop_recording() -> None
```

Stop recording data and publish the recorded data.

<a id="reachy_mini.daemon.backend.abstract.Backend.get_present_head_joint_positions"></a>

#### get\_present\_head\_joint\_positions

```python
def get_present_head_joint_positions(
) -> Annotated[NDArray[np.float64], (7, )]
```

Return the present head joint positions.

This method is a placeholder and should be overridden by subclasses.

<a id="reachy_mini.daemon.backend.abstract.Backend.get_present_body_yaw"></a>

#### get\_present\_body\_yaw

```python
def get_present_body_yaw() -> float
```

Return the present body yaw.

<a id="reachy_mini.daemon.backend.abstract.Backend.get_present_head_pose"></a>

#### get\_present\_head\_pose

```python
def get_present_head_pose() -> Annotated[NDArray[np.float64], (4, 4)]
```

Return the present head pose as a 4x4 matrix.

<a id="reachy_mini.daemon.backend.abstract.Backend.get_current_head_pose"></a>

#### get\_current\_head\_pose

```python
def get_current_head_pose() -> Annotated[NDArray[np.float64], (4, 4)]
```

Return the present head pose as a 4x4 matrix.

<a id="reachy_mini.daemon.backend.abstract.Backend.get_present_antenna_joint_positions"></a>

#### get\_present\_antenna\_joint\_positions

```python
def get_present_antenna_joint_positions(
) -> Annotated[NDArray[np.float64], (2, )]
```

Return the present antenna joint positions.

This method is a placeholder and should be overridden by subclasses.

<a id="reachy_mini.daemon.backend.abstract.Backend.update_head_kinematics_model"></a>

#### update\_head\_kinematics\_model

```python
def update_head_kinematics_model(
    head_joint_positions: Annotated[NDArray[np.float64], (7, )] | None = None,
    antennas_joint_positions: Annotated[NDArray[np.float64],
                                        (2, )] | None = None
) -> None
```

Update the placo kinematics of the robot.

**Arguments**:

- `head_joint_positions` _List[float] | None_ - The joint positions of the head.
- `antennas_joint_positions` _List[float] | None_ - The joint positions of the antennas.
  

**Returns**:

- `None` - This method does not return anything.
  
  This method updates the head kinematics model with the given joint positions.
  - If the joint positions are not provided, it will use the current joint positions.
  - If the head joint positions have not changed, it will return without recomputing the forward kinematics.
  - If the head joint positions have changed, it will compute the forward kinematics to get the current head pose.
  - If the forward kinematics fails, it will raise an assertion error.
  - If the antennas joint positions are provided, it will update the current antenna joint positions.
  

**Notes**:

  This method will update the `current_head_pose` and `current_head_joint_positions`
  attributes of the backend instance with the computed values. And the `current_antenna_joint_positions` if provided.

<a id="reachy_mini.daemon.backend.abstract.Backend.set_automatic_body_yaw"></a>

#### set\_automatic\_body\_yaw

```python
def set_automatic_body_yaw(body_yaw: float) -> None
```

Set the automatic body yaw.

**Arguments**:

- `body_yaw` _float_ - The yaw angle of the body.

<a id="reachy_mini.daemon.backend.abstract.Backend.get_urdf"></a>

#### get\_urdf

```python
def get_urdf() -> str
```

Get the URDF representation of the robot.

<a id="reachy_mini.daemon.backend.abstract.Backend.play_sound"></a>

#### play\_sound

```python
def play_sound(sound_file: str) -> None
```

Play a sound file from the assets directory.

If the file is not found in the assets directory, try to load the path itself.

**Arguments**:

- `sound_file` _str_ - The name of the sound file to play (e.g., "wake_up.wav").

<a id="reachy_mini.daemon.backend.abstract.Backend.wake_up"></a>

#### wake\_up

```python
async def wake_up() -> None
```

Wake up the robot - go to the initial head position and play the wake up emote and sound.

<a id="reachy_mini.daemon.backend.abstract.Backend.goto_sleep"></a>

#### goto\_sleep

```python
async def goto_sleep() -> None
```

Put the robot to sleep by moving the head and antennas to a predefined sleep position.

- If we are already very close to the sleep position, we do nothing.
- If we are far from the sleep position:
    - If we are far from the initial position, we move there first.
    - If we are close to the initial position, we move directly to the sleep position.

<a id="reachy_mini.daemon.backend.abstract.Backend.get_motor_control_mode"></a>

#### get\_motor\_control\_mode

```python
@abstractmethod
def get_motor_control_mode() -> MotorControlMode
```

Get the motor control mode.

<a id="reachy_mini.daemon.backend.abstract.Backend.set_motor_control_mode"></a>

#### set\_motor\_control\_mode

```python
@abstractmethod
def set_motor_control_mode(mode: MotorControlMode) -> None
```

Set the motor control mode.

<a id="reachy_mini.daemon.backend.abstract.Backend.get_present_passive_joint_positions"></a>

#### get\_present\_passive\_joint\_positions

```python
def get_present_passive_joint_positions() -> Optional[Dict[str, float]]
```

Get the present passive joint positions.

Requires the Placo kinematics engine.

<a id="reachy_mini.daemon.backend.mujoco.utils"></a>

# reachy\_mini.daemon.backend.mujoco.utils

Mujoco utilities for Reachy Mini.

This module provides utility functions for working with MuJoCo models, including
homogeneous transformation matrices, joint positions, and actuator names.

<a id="reachy_mini.daemon.backend.mujoco.utils.get_homogeneous_matrix_from_euler"></a>

#### get\_homogeneous\_matrix\_from\_euler

```python
def get_homogeneous_matrix_from_euler(position: tuple[float, float,
                                                      float] = (0, 0, 0),
                                      euler_angles: tuple[float, float,
                                                          float] = (0, 0, 0),
                                      degrees: bool = False) -> Annotated[
                                          npt.NDArray[np.float64], (4, 4)]
```

Return a homogeneous transformation matrix from position and Euler angles.

<a id="reachy_mini.daemon.backend.mujoco.utils.get_joint_qpos"></a>

#### get\_joint\_qpos

```python
def get_joint_qpos(model: MjModel, data: MjData, joint_name: str) -> float
```

Return the qpos (rad) of a specified joint in the model.

<a id="reachy_mini.daemon.backend.mujoco.utils.get_joint_id_from_name"></a>

#### get\_joint\_id\_from\_name

```python
def get_joint_id_from_name(model: MjModel, name: str) -> int
```

Return the id of a specified joint.

<a id="reachy_mini.daemon.backend.mujoco.utils.get_joint_addr_from_name"></a>

#### get\_joint\_addr\_from\_name

```python
def get_joint_addr_from_name(model: MjModel, name: str) -> int
```

Return the address of a specified joint.

<a id="reachy_mini.daemon.backend.mujoco.utils.get_actuator_names"></a>

#### get\_actuator\_names

```python
def get_actuator_names(model: MjModel) -> list[str]
```

Return the list of the actuators names from the MuJoCo model.

<a id="reachy_mini.daemon.backend.mujoco.backend"></a>

# reachy\_mini.daemon.backend.mujoco.backend

Mujoco Backend for Reachy Mini.

This module provides the MujocoBackend class for simulating the Reachy Mini robot using the MuJoCo physics engine.

It includes methods for running the simulation, getting joint positions, and controlling the robot's joints.

<a id="reachy_mini.daemon.backend.mujoco.backend.MujocoBackend"></a>

## MujocoBackend Objects

```python
class MujocoBackend(Backend)
```

Simulated Reachy Mini using MuJoCo.

<a id="reachy_mini.daemon.backend.mujoco.backend.MujocoBackend.__init__"></a>

#### \_\_init\_\_

```python
def __init__(scene: str = "empty",
             check_collision: bool = False,
             kinematics_engine: str = "AnalyticalKinematics",
             headless: bool = False) -> None
```

Initialize the MujocoBackend with a specified scene.

**Arguments**:

- `scene` _str_ - The name of the scene to load. Default is "empty".
- `check_collision` _bool_ - If True, enable collision checking. Default is False.
- `kinematics_engine` _str_ - Kinematics engine to use. Defaults to "AnalyticalKinematics".
- `headless` _bool_ - If True, run Mujoco in headless mode (no GUI). Default is False.

<a id="reachy_mini.daemon.backend.mujoco.backend.MujocoBackend.rendering_loop"></a>

#### rendering\_loop

```python
def rendering_loop() -> None
```

Offline Rendering loop for the Mujoco simulation.

Capture the image from the virtual Reachy's camera and send it over UDP.

<a id="reachy_mini.daemon.backend.mujoco.backend.MujocoBackend.run"></a>

#### run

```python
def run() -> None
```

Run the Mujoco simulation with a viewer.

This method initializes the viewer and enters the main simulation loop.
It updates the joint positions at a rate and publishes the joint positions.

<a id="reachy_mini.daemon.backend.mujoco.backend.MujocoBackend.get_mj_present_head_pose"></a>

#### get\_mj\_present\_head\_pose

```python
def get_mj_present_head_pose() -> Annotated[npt.NDArray[np.float64], (4, 4)]
```

Get the current head pose from the Mujoco simulation.

**Returns**:

- `np.ndarray` - The current head pose as a 4x4 transformation matrix.

<a id="reachy_mini.daemon.backend.mujoco.backend.MujocoBackend.close"></a>

#### close

```python
def close() -> None
```

Close the Mujoco backend.

<a id="reachy_mini.daemon.backend.mujoco.backend.MujocoBackend.get_status"></a>

#### get\_status

```python
def get_status() -> "MujocoBackendStatus"
```

Get the status of the Mujoco backend.

**Returns**:

- `dict` - An empty dictionary as the Mujoco backend does not have a specific status to report.

<a id="reachy_mini.daemon.backend.mujoco.backend.MujocoBackend.get_present_head_joint_positions"></a>

#### get\_present\_head\_joint\_positions

```python
def get_present_head_joint_positions(
) -> Annotated[npt.NDArray[np.float64], (7, )]
```

Get the current joint positions of the head.

<a id="reachy_mini.daemon.backend.mujoco.backend.MujocoBackend.get_present_antenna_joint_positions"></a>

#### get\_present\_antenna\_joint\_positions

```python
def get_present_antenna_joint_positions(
) -> Annotated[npt.NDArray[np.float64], (2, )]
```

Get the current joint positions of the antennas.

<a id="reachy_mini.daemon.backend.mujoco.backend.MujocoBackend.get_motor_control_mode"></a>

#### get\_motor\_control\_mode

```python
def get_motor_control_mode() -> MotorControlMode
```

Get the motor control mode.

<a id="reachy_mini.daemon.backend.mujoco.backend.MujocoBackend.set_motor_control_mode"></a>

#### set\_motor\_control\_mode

```python
def set_motor_control_mode(mode: MotorControlMode) -> None
```

Set the motor control mode.

<a id="reachy_mini.daemon.backend.mujoco.backend.MujocoBackendStatus"></a>

## MujocoBackendStatus Objects

```python
@dataclass
class MujocoBackendStatus()
```

Dataclass to represent the status of the Mujoco backend.

Empty for now, as the Mujoco backend does not have a specific status to report.

<a id="reachy_mini.daemon.backend.mujoco.video_udp"></a>

# reachy\_mini.daemon.backend.mujoco.video\_udp

UDP JPEG Frame Sender.

This module provides a class to send JPEG frames over UDP. It encodes the frames as JPEG images and splits them into chunks to fit within the maximum packet size for UDP transmission.

<a id="reachy_mini.daemon.backend.mujoco.video_udp.UDPJPEGFrameSender"></a>

## UDPJPEGFrameSender Objects

```python
class UDPJPEGFrameSender()
```

A class to send JPEG frames over UDP.

<a id="reachy_mini.daemon.backend.mujoco.video_udp.UDPJPEGFrameSender.__init__"></a>

#### \_\_init\_\_

```python
def __init__(dest_ip: str = "127.0.0.1",
             dest_port: int = 5005,
             max_packet_size: int = 1400) -> None
```

Initialize the UDPJPEGFrameSender.

**Arguments**:

- `dest_ip` _str_ - Destination IP address.
- `dest_port` _int_ - Destination port number.
- `max_packet_size` _int_ - Maximum size of each UDP packet.

<a id="reachy_mini.daemon.backend.mujoco.video_udp.UDPJPEGFrameSender.send_frame"></a>

#### send\_frame

```python
def send_frame(frame: npt.NDArray[np.uint8]) -> None
```

Send a frame as a JPEG image over UDP.

**Arguments**:

- `frame` _np.ndarray_ - The frame to be sent, in RGB format.

<a id="reachy_mini.daemon.backend.mujoco"></a>

# reachy\_mini.daemon.backend.mujoco

MuJoCo Backend for Reachy Mini Daemon.

<a id="reachy_mini.daemon.backend.robot.backend"></a>

# reachy\_mini.daemon.backend.robot.backend

Robot Backend for Reachy Mini.

This module provides the `RobotBackend` class, which interfaces with the Reachy Mini motor controller to control the robot's movements and manage its status.
It handles the control loop, joint positions, torque enabling/disabling, and provides a status report of the robot's backend.
It uses the `ReachyMiniMotorController` to communicate with the robot's motors.

<a id="reachy_mini.daemon.backend.robot.backend.RobotBackend"></a>

## RobotBackend Objects

```python
class RobotBackend(Backend)
```

Real robot backend for Reachy Mini.

<a id="reachy_mini.daemon.backend.robot.backend.RobotBackend.__init__"></a>

#### \_\_init\_\_

```python
def __init__(serialport: str,
             log_level: str = "INFO",
             check_collision: bool = False,
             kinematics_engine: str = "AnalyticalKinematics",
             hardware_error_check_frequency: float = 1.0)
```

Initialize the RobotBackend.

**Arguments**:

- `serialport` _str_ - The serial port to which the Reachy Mini is connected.
- `log_level` _str_ - The logging level for the backend. Default is "INFO".
- `check_collision` _bool_ - If True, enable collision checking. Default is False.
- `kinematics_engine` _str_ - Kinematics engine to use. Defaults to "AnalyticalKinematics".
- `hardware_error_check_frequency` _float_ - Frequency in seconds to check for hardware errors. Default is 1.0.
  
  Tries to connect to the Reachy Mini motor controller and initializes the control loop.

<a id="reachy_mini.daemon.backend.robot.backend.RobotBackend.run"></a>

#### run

```python
def run() -> None
```

Run the control loop for the robot backend.

This method continuously updates the motor controller at a specified frequency.
It reads the joint positions, updates the motor controller, and publishes the joint positions.
It also handles errors and retries if the motor controller is not responding.

<a id="reachy_mini.daemon.backend.robot.backend.RobotBackend.close"></a>

#### close

```python
def close() -> None
```

Close the motor controller connection.

<a id="reachy_mini.daemon.backend.robot.backend.RobotBackend.get_status"></a>

#### get\_status

```python
def get_status() -> "RobotBackendStatus"
```

Get the current status of the robot backend.

<a id="reachy_mini.daemon.backend.robot.backend.RobotBackend.enable_motors"></a>

#### enable\_motors

```python
def enable_motors() -> None
```

Enable the motors by turning the torque on.

<a id="reachy_mini.daemon.backend.robot.backend.RobotBackend.disable_motors"></a>

#### disable\_motors

```python
def disable_motors() -> None
```

Disable the motors by turning the torque off.

<a id="reachy_mini.daemon.backend.robot.backend.RobotBackend.set_head_operation_mode"></a>

#### set\_head\_operation\_mode

```python
def set_head_operation_mode(mode: int) -> None
```

Change the operation mode of the head motors.

**Arguments**:

- `mode` _int_ - The operation mode for the head motors.
  
  The operation modes can be:
- `0` - torque control
- `3` - position control
- `5` - current-based position control.
  
  Important:
  This method does not work well with the current feetech motors (body rotation), as they do not support torque control.
  So the method disables the antennas when in torque control mode.
  The dynamixel motors used for the head do support torque control, so this method works as expected.
  

**Arguments**:

- `mode` _int_ - The operation mode for the head motors.
  This could be a specific mode like position control, velocity control, or torque control.

<a id="reachy_mini.daemon.backend.robot.backend.RobotBackend.set_antennas_operation_mode"></a>

#### set\_antennas\_operation\_mode

```python
def set_antennas_operation_mode(mode: int) -> None
```

Change the operation mode of the antennas motors.

**Arguments**:

- `mode` _int_ - The operation mode for the antennas motors (0: torque control, 3: position control, 5: current-based position control).
  
  Important:
  This method does not work well with the current feetech motors, as they do not support torque control.
  So the method disables the antennas when in torque control mode.
  

**Arguments**:

- `mode` _int_ - The operation mode for the antennas motors.
  This could be a specific mode like position control, velocity control, or torque control.

<a id="reachy_mini.daemon.backend.robot.backend.RobotBackend.get_all_joint_positions"></a>

#### get\_all\_joint\_positions

```python
def get_all_joint_positions() -> tuple[list[float], list[float]]
```

Get the current joint positions of the robot.

**Returns**:

- `tuple` - A tuple containing two lists - the first list is for the head joint positions,
  and the second list is for the antenna joint positions.

<a id="reachy_mini.daemon.backend.robot.backend.RobotBackend.get_present_head_joint_positions"></a>

#### get\_present\_head\_joint\_positions

```python
def get_present_head_joint_positions(
) -> Annotated[npt.NDArray[np.float64], (7, )]
```

Get the current joint positions of the head.

**Returns**:

- `list` - A list of joint positions for the head, including the body rotation.

<a id="reachy_mini.daemon.backend.robot.backend.RobotBackend.get_present_antenna_joint_positions"></a>

#### get\_present\_antenna\_joint\_positions

```python
def get_present_antenna_joint_positions(
) -> Annotated[npt.NDArray[np.float64], (2, )]
```

Get the current joint positions of the antennas.

**Returns**:

- `list` - A list of joint positions for the antennas.

<a id="reachy_mini.daemon.backend.robot.backend.RobotBackend.compensate_head_gravity"></a>

#### compensate\_head\_gravity

```python
def compensate_head_gravity() -> None
```

Calculate the currents necessary to compensate for gravity.

<a id="reachy_mini.daemon.backend.robot.backend.RobotBackend.get_motor_control_mode"></a>

#### get\_motor\_control\_mode

```python
def get_motor_control_mode() -> MotorControlMode
```

Get the motor control mode.

<a id="reachy_mini.daemon.backend.robot.backend.RobotBackend.set_motor_control_mode"></a>

#### set\_motor\_control\_mode

```python
def set_motor_control_mode(mode: MotorControlMode) -> None
```

Set the motor control mode.

<a id="reachy_mini.daemon.backend.robot.backend.RobotBackend.read_hardware_errors"></a>

#### read\_hardware\_errors

```python
def read_hardware_errors() -> dict[str, list[str]]
```

Read hardware errors from the motor controller.

<a id="reachy_mini.daemon.backend.robot.backend.RobotBackendStatus"></a>

## RobotBackendStatus Objects

```python
@dataclass
class RobotBackendStatus()
```

Status of the Robot Backend.

<a id="reachy_mini.daemon.backend.robot"></a>

# reachy\_mini.daemon.backend.robot

Real robot backend for Reachy Mini.

<a id="reachy_mini.daemon.backend"></a>

# reachy\_mini.daemon.backend

Backend module for Reachy Mini Daemon.

<a id="reachy_mini.daemon.daemon"></a>

# reachy\_mini.daemon.daemon

Daemon for Reachy Mini robot.

This module provides a daemon that runs a backend for either a simulated Reachy Mini using Mujoco or a real Reachy Mini robot using a serial connection.
It includes methods to start, stop, and restart the daemon, as well as to check its status.
It also provides a command-line interface for easy interaction.

<a id="reachy_mini.daemon.daemon.Daemon"></a>

## Daemon Objects

```python
class Daemon()
```

Daemon for simulated or real Reachy Mini robot.

Runs the server with the appropriate backend (Mujoco for simulation or RobotBackend for real hardware).

<a id="reachy_mini.daemon.daemon.Daemon.__init__"></a>

#### \_\_init\_\_

```python
def __init__(log_level: str = "INFO", wireless_version: bool = False) -> None
```

Initialize the Reachy Mini daemon.

<a id="reachy_mini.daemon.daemon.Daemon.start"></a>

#### start

```python
async def start(sim: bool = False,
                serialport: str = "auto",
                scene: str = "empty",
                localhost_only: bool = True,
                wake_up_on_start: bool = True,
                check_collision: bool = False,
                kinematics_engine: str = "AnalyticalKinematics",
                headless: bool = False) -> "DaemonState"
```

Start the Reachy Mini daemon.

**Arguments**:

- `sim` _bool_ - If True, run in simulation mode using Mujoco. Defaults to False.
- `serialport` _str_ - Serial port for real motors. Defaults to "auto", which will try to find the port automatically.
- `scene` _str_ - Name of the scene to load in simulation mode ("empty" or "minimal"). Defaults to "empty".
- `localhost_only` _bool_ - If True, restrict the server to localhost only clients. Defaults to True.
- `wake_up_on_start` _bool_ - If True, wake up Reachy Mini on start. Defaults to True.
- `check_collision` _bool_ - If True, enable collision checking. Defaults to False.
- `kinematics_engine` _str_ - Kinematics engine to use. Defaults to "AnalyticalKinematics".
- `headless` _bool_ - If True, run Mujoco in headless mode (no GUI). Defaults to False.
  

**Returns**:

- `DaemonState` - The current state of the daemon after attempting to start it.

<a id="reachy_mini.daemon.daemon.Daemon.stop"></a>

#### stop

```python
async def stop(goto_sleep_on_stop: bool = True) -> "DaemonState"
```

Stop the Reachy Mini daemon.

**Arguments**:

- `goto_sleep_on_stop` _bool_ - If True, put Reachy Mini to sleep on stop. Defaults to True.
  

**Returns**:

- `DaemonState` - The current state of the daemon after attempting to stop it.

<a id="reachy_mini.daemon.daemon.Daemon.restart"></a>

#### restart

```python
async def restart(sim: Optional[bool] = None,
                  serialport: Optional[str] = None,
                  scene: Optional[str] = None,
                  headless: Optional[bool] = None,
                  localhost_only: Optional[bool] = None,
                  wake_up_on_start: Optional[bool] = None,
                  goto_sleep_on_stop: Optional[bool] = None) -> "DaemonState"
```

Restart the Reachy Mini daemon.

**Arguments**:

- `sim` _bool_ - If True, run in simulation mode using Mujoco. Defaults to None (uses the previous value).
- `serialport` _str_ - Serial port for real motors. Defaults to None (uses the previous value).
- `scene` _str_ - Name of the scene to load in simulation mode ("empty" or "minimal"). Defaults to None (uses the previous value).
- `headless` _bool_ - If True, run Mujoco in headless mode (no GUI). Defaults to None (uses the previous value).
- `localhost_only` _bool_ - If True, restrict the server to localhost only clients. Defaults to None (uses the previous value).
- `wake_up_on_start` _bool_ - If True, wake up Reachy Mini on start. Defaults to None (don't wake up).
- `goto_sleep_on_stop` _bool_ - If True, put Reachy Mini to sleep on stop. Defaults to None (don't go to sleep).
  

**Returns**:

- `DaemonState` - The current state of the daemon after attempting to restart it.

<a id="reachy_mini.daemon.daemon.Daemon.status"></a>

#### status

```python
def status() -> "DaemonStatus"
```

Get the current status of the Reachy Mini daemon.

<a id="reachy_mini.daemon.daemon.Daemon.run4ever"></a>

#### run4ever

```python
async def run4ever(sim: bool = False,
                   serialport: str = "auto",
                   scene: str = "empty",
                   localhost_only: bool = True,
                   wake_up_on_start: bool = True,
                   goto_sleep_on_stop: bool = True,
                   check_collision: bool = False,
                   kinematics_engine: str = "AnalyticalKinematics",
                   headless: bool = False) -> None
```

Run the Reachy Mini daemon indefinitely.

First, it starts the daemon, then it keeps checking the status and allows for graceful shutdown on user interrupt (Ctrl+C).

**Arguments**:

- `sim` _bool_ - If True, run in simulation mode using Mujoco. Defaults to False.
- `serialport` _str_ - Serial port for real motors. Defaults to "auto", which will try to find the port automatically.
- `scene` _str_ - Name of the scene to load in simulation mode ("empty" or "minimal"). Defaults to "empty".
- `localhost_only` _bool_ - If True, restrict the server to localhost only clients. Defaults to True.
- `wake_up_on_start` _bool_ - If True, wake up Reachy Mini on start. Defaults to True.
- `goto_sleep_on_stop` _bool_ - If True, put Reachy Mini to sleep on stop. Defaults to True
- `check_collision` _bool_ - If True, enable collision checking. Defaults to False.
- `kinematics_engine` _str_ - Kinematics engine to use. Defaults to "AnalyticalKinematics".
- `headless` _bool_ - If True, run Mujoco in headless mode (no GUI). Defaults to False.

<a id="reachy_mini.daemon.daemon.DaemonState"></a>

## DaemonState Objects

```python
class DaemonState(Enum)
```

Enum representing the state of the Reachy Mini daemon.

<a id="reachy_mini.daemon.daemon.DaemonStatus"></a>

## DaemonStatus Objects

```python
@dataclass
class DaemonStatus()
```

Dataclass representing the status of the Reachy Mini daemon.

<a id="reachy_mini.daemon"></a>

# reachy\_mini.daemon

Daemon for Reachy Mini.

<a id="reachy_mini.motion.move"></a>

# reachy\_mini.motion.move

Module for defining motion moves on the ReachyMini robot.

<a id="reachy_mini.motion.move.Move"></a>

## Move Objects

```python
class Move(ABC)
```

Abstract base class for defining a move on the ReachyMini robot.

<a id="reachy_mini.motion.move.Move.duration"></a>

#### duration

```python
@property
@abstractmethod
def duration() -> float
```

Duration of the move in seconds.

<a id="reachy_mini.motion.move.Move.evaluate"></a>

#### evaluate

```python
@abstractmethod
def evaluate(
    t: float
) -> tuple[npt.NDArray[np.float64] | None, npt.NDArray[np.float64] | None,
           float | None]
```

Evaluate the move at time t, typically called at a high-frequency (eg. 100Hz).

**Arguments**:

- `t` - The time at which to evaluate the move (in seconds). It will always be between 0 and duration.
  

**Returns**:

- `head` - The head position (4x4 homogeneous matrix).
- `antennas` - The antennas positions (rad).
- `body_yaw` - The body yaw angle (rad).

<a id="reachy_mini.motion.goto"></a>

# reachy\_mini.motion.goto

A goto move to a target head pose and/or antennas position.

<a id="reachy_mini.motion.goto.GotoMove"></a>

## GotoMove Objects

```python
class GotoMove(Move)
```

A goto move to a target head pose and/or antennas position.

<a id="reachy_mini.motion.goto.GotoMove.__init__"></a>

#### \_\_init\_\_

```python
def __init__(start_head_pose: npt.NDArray[np.float64],
             target_head_pose: npt.NDArray[np.float64] | None,
             start_antennas: npt.NDArray[np.float64],
             target_antennas: npt.NDArray[np.float64] | None,
             start_body_yaw: float, target_body_yaw: float | None,
             duration: float, method: InterpolationTechnique)
```

Set up the goto move.

<a id="reachy_mini.motion.goto.GotoMove.duration"></a>

#### duration

```python
@property
def duration() -> float
```

Duration of the goto in seconds.

<a id="reachy_mini.motion.goto.GotoMove.evaluate"></a>

#### evaluate

```python
def evaluate(
    t: float
) -> tuple[npt.NDArray[np.float64] | None, npt.NDArray[np.float64] | None,
           float | None]
```

Evaluate the goto at time t.

<a id="reachy_mini.motion.recorded_move"></a>

# reachy\_mini.motion.recorded\_move

<a id="reachy_mini.motion.recorded_move.lerp"></a>

#### lerp

```python
def lerp(v0: float, v1: float, alpha: float) -> float
```

Linear interpolation between two values.

<a id="reachy_mini.motion.recorded_move.RecordedMove"></a>

## RecordedMove Objects

```python
class RecordedMove(Move)
```

Represent a recorded move.

<a id="reachy_mini.motion.recorded_move.RecordedMove.__init__"></a>

#### \_\_init\_\_

```python
def __init__(move: Dict[str, Any]) -> None
```

Initialize RecordedMove.

<a id="reachy_mini.motion.recorded_move.RecordedMove.duration"></a>

#### duration

```python
@property
def duration() -> float
```

Get the duration of the recorded move.

<a id="reachy_mini.motion.recorded_move.RecordedMove.evaluate"></a>

#### evaluate

```python
def evaluate(
    t: float
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float]
```

Evaluate the move at time t.

**Returns**:

- `head` - The head position (4x4 homogeneous matrix).
- `antennas` - The antennas positions (rad).
- `body_yaw` - The body yaw angle (rad).

<a id="reachy_mini.motion.recorded_move.RecordedMoves"></a>

## RecordedMoves Objects

```python
class RecordedMoves()
```

Load a library of recorded moves from a HuggingFace dataset.

<a id="reachy_mini.motion.recorded_move.RecordedMoves.__init__"></a>

#### \_\_init\_\_

```python
def __init__(hf_dataset_name: str)
```

Initialize RecordedMoves.

<a id="reachy_mini.motion.recorded_move.RecordedMoves.process"></a>

#### process

```python
def process() -> None
```

Populate recorded moves and sounds.

<a id="reachy_mini.motion.recorded_move.RecordedMoves.get"></a>

#### get

```python
def get(move_name: str) -> RecordedMove
```

Get a recorded move by name.

<a id="reachy_mini.motion.recorded_move.RecordedMoves.list_moves"></a>

#### list\_moves

```python
def list_moves() -> List[str]
```

List all moves in the loaded library.

<a id="reachy_mini.motion"></a>

# reachy\_mini.motion

Motion module for Reachy Mini.

This module contains both utilities to create and play moves, as well as utilities to download datasets of recorded moves.


# Examples


## body_yaw_test.py

"""Reachy Mini Head Position GUI Example."""

import time

import numpy as np
from scipy.spatial.transform import Rotation as R

from reachy_mini import ReachyMini


def main():
    """Run a GUI to set the head position and orientation of Reachy Mini."""
    with ReachyMini() as mini:
        # with ReachyMini(automatic_body_yaw=False) as mini:
        t0 = time.time()

        while True:
            t = time.time() - t0
            target = np.deg2rad(90) * np.sin(2 * np.pi * 0.5 * t)

            head = np.eye(4)
            head[:3, 3] = [0, 0, 0]

            # Read values from the GUI
            roll = np.deg2rad(0.0)
            pitch = np.deg2rad(0.0)
            yaw = np.deg2rad(0.0)
            head[:3, :3] = R.from_euler(
                "xyz", [roll, pitch, yaw], degrees=False
            ).as_matrix()

            mini.set_target(
                head=head,
                antennas=np.array([target, -target]),
                body_yaw=target,
            )
            time.sleep(0.01)


if __name__ == "__main__":
    main()

## compare_placo_nn_kin.py

import os  # noqa: D100

import numpy as np
from placo_utils.tf import tf

from reachy_mini.kinematics import NNKinematics, PlacoKinematics

urdf_path = os.path.abspath(
    "../../src/reachy_mini/descriptions/reachy_mini/urdf/robot.urdf"
)

placo_kinematics = PlacoKinematics(urdf_path)
placo_kinematics.robot.update_kinematics()
nn_kinematics = NNKinematics("../../src/reachy_mini/assets/models/")

i = -1
while i < 2000:
    i += 1
    px, py, pz = [np.random.uniform(-0.01, 0.01) for _ in range(3)]
    roll, pitch = [np.random.uniform(-np.deg2rad(30), np.deg2rad(30)) for _ in range(2)]
    yaw = np.random.uniform(-2.8, 2.8)
    # yaw = 0
    body_yaw = -yaw  # + np.random.uniform(-np.deg2rad(20), np.deg2rad(20))
    body_yaw = 0

    T_head_target = tf.translation_matrix((px, py, pz)) @ tf.euler_matrix(
        roll, pitch, yaw
    )

    placo_result = placo_kinematics.ik(
        pose=T_head_target, body_yaw=body_yaw, no_iterations=20
    )
    nn_result = nn_kinematics.ik(pose=T_head_target, body_yaw=body_yaw)

    print(f"Placo Kinematics Result: {np.around(placo_result, 3)}")
    print(f"NN Kinematics Result: {np.around(nn_result, 3)}")
    print("==")

## compare_recordings.py

#!/usr/bin/env python3
# compare_recordings.py
"""Compare two ReachyMini dance-run recordings by regenerating overlay plots.

Usage
-----
python compare_recordings.py measures/2025-08-07_11-36-00 measures/2025-08-27_15-30-30

What it does
------------
- Scans both input folders for per-move .npz files (created by your tracker).
- Keeps only the intersection of move names present in BOTH folders.
- For each common move:
  1) Recreates the 3-row error stack (translation [mm], angular [deg], combined [mm])
     and overlays the two runs.
  2) Recreates the 6-row XYZ/RPY stack. It overlays the "present" trajectories from
     both runs and shows the "goal" from run A as a thin reference line.
- Time axes are normalized per run (t - t[0]) and simply overlaid (no resampling).
- Saves results into: measures/compare_<runA>_vs_<runB>/

Notes
-----
- Expects the .npz schema produced by your data script:
  keys: t, trans_mm, ang_deg, magic_mm, goal_pos_m, present_pos_m,
        goal_rpy_deg, present_rpy_deg
- If a key is missing for any move in either run, that move is skipped with a warning.
- Matplotlib only, no seaborn. Plots use grids and legends; sizes chosen for readability.

Dependencies: numpy, matplotlib

"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, Optional, Set

import matplotlib.pyplot as plt
import numpy as np


# ------------------------- Logging ---------------------------------
def setup_logging() -> None:  # noqa: D103
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


# ------------------------- I/O helpers ------------------------------
NPZ_REQUIRED_KEYS = {
    "t",
    "trans_mm",
    "ang_deg",
    "magic_mm",
    "goal_pos_m",
    "present_pos_m",
    "goal_rpy_deg",
    "present_rpy_deg",
}


def list_moves(folder: Path) -> Set[str]:
    """Return the set of move basenames (without extension) for all .npz files."""
    return {p.stem for p in folder.glob("*.npz") if p.is_file()}


def load_npz_safe(path: Path) -> Optional[Dict[str, np.ndarray]]:
    """Load a .npz file and validate required keys. Returns dict or None."""
    try:
        with np.load(path) as data:
            keys = set(data.files)
            missing = NPZ_REQUIRED_KEYS - keys
            if missing:
                logging.warning("File %s missing keys: %s", path, sorted(missing))
                return None
            return {k: data[k] for k in NPZ_REQUIRED_KEYS}
    except Exception as e:
        logging.warning("Failed to load %s: %s", path, e)
        return None


# ------------------------- Plotting --------------------------------
def plot_errors_compare(
    A: Dict[str, np.ndarray],
    B: Dict[str, np.ndarray],
    move_name: str,
    out_png: Path,
) -> None:
    """Overlay error stacks (translation, angular, combined) for two runs."""
    tA = A["t"] - A["t"][0]
    tB = B["t"] - B["t"][0]

    fig, axes = plt.subplots(
        nrows=3, ncols=1, sharex=False, figsize=(12, 9), constrained_layout=True
    )

    ax = axes[0]
    ax.plot(tA, A["trans_mm"], linewidth=1.6, label="A trans_mm")
    ax.plot(tB, B["trans_mm"], linewidth=1.6, label="B trans_mm")
    ax.set_ylabel("Position error [mm]")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    ax = axes[1]
    ax.plot(tA, A["ang_deg"], linewidth=1.6, label="A ang_deg")
    ax.plot(tB, B["ang_deg"], linewidth=1.6, label="B ang_deg")
    ax.set_ylabel("Angular error [deg]")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    ax = axes[2]
    ax.plot(tA, A["magic_mm"], linewidth=1.6, label="A combined")
    ax.plot(tB, B["magic_mm"], linewidth=1.6, label="B combined")
    ax.set_ylabel("Combined [magic-mm]")
    ax.set_xlabel("Time [s] (each run normalized to start at 0)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    fig.suptitle(
        f"Pose tracking errors vs time – compare A vs B – {move_name}", fontsize=14
    )
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_xyzrpy_compare(
    A: Dict[str, np.ndarray],
    B: Dict[str, np.ndarray],
    move_name: str,
    out_png: Path,
) -> None:
    """Overlay XYZ (mm) and RPY (deg) present trajectories for two runs.

    Also plot goal from run A as a thin reference line.
    """
    tA = A["t"] - A["t"][0]
    tB = B["t"] - B["t"][0]

    # Positions in mm
    goal_pos_A_mm = A["goal_pos_m"] * 1000.0
    present_A_mm = A["present_pos_m"] * 1000.0
    present_B_mm = B["present_pos_m"] * 1000.0

    # RPY in deg
    goal_rpy_A_deg = A["goal_rpy_deg"]
    present_A_rpy = A["present_rpy_deg"]
    present_B_rpy = B["present_rpy_deg"]

    labels = [
        ("X position [mm]", 0),
        ("Y position [mm]", 1),
        ("Z position [mm]", 2),
        ("Roll [deg]", 0),
        ("Pitch [deg]", 1),
        ("Yaw [deg]", 2),
    ]

    fig, axes = plt.subplots(
        nrows=6, ncols=1, sharex=False, figsize=(12, 14), constrained_layout=True
    )

    # XYZ: goal(A) thin, present(A), present(B)
    for ax, (ylabel, idx) in zip(axes[:3], labels[:3]):
        ax.plot(tA, goal_pos_A_mm[:, idx], linewidth=0.8, label=f"goal_A_{'xyz'[idx]}")
        ax.plot(
            tA, present_A_mm[:, idx], linewidth=1.6, label=f"present_A_{'xyz'[idx]}"
        )
        ax.plot(
            tB, present_B_mm[:, idx], linewidth=1.6, label=f"present_B_{'xyz'[idx]}"
        )
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()

    # RPY: goal(A) thin, present(A), present(B)
    rpy_names = ["roll", "pitch", "yaw"]
    for ax, (ylabel, idx) in zip(axes[3:], labels[3:]):
        ax.plot(
            tA, goal_rpy_A_deg[:, idx], linewidth=0.8, label=f"goal_A_{rpy_names[idx]}"
        )
        ax.plot(
            tA,
            present_A_rpy[:, idx],
            linewidth=1.6,
            label=f"present_A_{rpy_names[idx]}",
        )
        ax.plot(
            tB,
            present_B_rpy[:, idx],
            linewidth=1.6,
            label=f"present_B_{rpy_names[idx]}",
        )
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()

    axes[-1].set_xlabel("Time [s] (each run normalized to start at 0)")
    fig.suptitle(
        f"Head XYZ (mm) and RPY (deg) vs time – compare A vs B – {move_name}",
        fontsize=14,
    )
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


# ------------------------- Orchestration ----------------------------
def derive_output_root(dirA: Path, dirB: Path) -> Path:
    """Create output folder under the common 'measures' parent."""
    nameA = dirA.name
    nameB = dirB.name
    parentA = dirA.parent
    parentB = dirB.parent
    # Prefer parent of A if both look like 'measures'
    measures_parent = parentA if parentA.name == "measures" else parentA
    if parentB == parentA:
        measures_parent = parentA
    out = measures_parent / f"compare_{nameA}_vs_{nameB}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def process_move(move: str, dirA: Path, dirB: Path, out_dir: Path) -> None:  # noqa: D103
    pathA = dirA / f"{move}.npz"
    pathB = dirB / f"{move}.npz"

    dataA = load_npz_safe(pathA)
    dataB = load_npz_safe(pathB)
    if dataA is None or dataB is None:
        logging.warning("Skipping move '%s' due to load/keys issue.", move)
        return

    # Errors overlay
    out_err = out_dir / f"{move}_errors_compare.png"
    plot_errors_compare(dataA, dataB, move_name=move, out_png=out_err)

    # XYZ/RPY overlay
    out_xyzrpy = out_dir / f"{move}_xyzrpy_compare.png"
    plot_xyzrpy_compare(dataA, dataB, move_name=move, out_png=out_xyzrpy)

    logging.info("Saved %s and %s", out_err, out_xyzrpy)


def main() -> None:  # noqa: D103
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Regenerate and compare per-move plots from two recordings."
    )
    parser.add_argument(
        "dirA", type=Path, help="First run folder (e.g., measures/2025-08-07_11-36-00)"
    )
    parser.add_argument(
        "dirB", type=Path, help="Second run folder (e.g., measures/2025-08-27_15-30-30)"
    )
    args = parser.parse_args()

    dirA: Path = args.dirA
    dirB: Path = args.dirB

    if not dirA.is_dir() or not dirB.is_dir():
        logging.error("Both arguments must be existing directories.")
        return

    movesA = list_moves(dirA)
    movesB = list_moves(dirB)
    common = sorted(movesA & movesB)
    if not common:
        logging.error("No common moves found between %s and %s.", dirA, dirB)
        return

    out_dir = derive_output_root(dirA, dirB)
    logging.info("Common moves: %d. Output: %s", len(common), out_dir)

    for move in common:
        process_move(move, dirA, dirB, out_dir)

    logging.info("Done. Compared %d moves.", len(common))


if __name__ == "__main__":
    main()

## gravity_compensation_direct_control.py

"""Reachy Mini Gravity Compensation Direct Control Example."""

import time

import numpy as np
from placo_utils.visualization import robot_viz
from reachy_mini_motor_controller import ReachyMiniMotorController

from reachy_mini.kinematics import PlacoKinematics


def main():
    """Run a demo to compensate the gravity of the Reachy Mini platform."""
    urdf_path = "src/reachy_mini/descriptions/reachy_mini/urdf/robot.urdf"
    solver = PlacoKinematics(urdf_path, 0.02)
    robot = solver.robot
    robot.update_kinematics()
    viz = robot_viz(robot)

    # Initialize the motor controller (adjust port if needed)
    controller = ReachyMiniMotorController(serialport="/dev/ttyACM0")

    # Details found here in the Specifications table
    # https://emanual.robotis.com/docs/en/dxl/x/xl330-m288/#Specifications
    # the torque constant seems to be nonlinear and is not constant!!!!
    k_Nm_to_mA = (
        1.47 / 0.52 * 1000
    )  # Conversion factor from Nm to mA for the Stewart platform motors
    efficiency = 1.0  # Efficiency of the motors
    # torque constant correction factor
    correction_factor = 3.0  # This number is valid for currents under 30mA

    t0 = time.time()
    controller.disable_torque()  # Disable torque for the Stewart platform motors
    controller.set_stewart_platform_operating_mode(
        0
    )  # Set operation mode to torque control
    controller.enable_torque()  # Enable torque for the Stewart platform motors

    print("Robot is now compliant. Press Ctrl+C to exit.")
    try:
        while time.time() - t0 < 150.0:  # Wait for the motors to stabilize
            motor_pos = controller.read_all_positions()
            head_pos = [motor_pos[0]] + motor_pos[
                3:
            ]  # Extract head motor positions (all_yaw, 1, 2, 3, 4, 5, 6)

            # compute the gravity torque
            gravity_torque = solver.compute_gravity_torque(head_pos)
            # the target motor current
            current = gravity_torque * k_Nm_to_mA / efficiency / correction_factor  # mA
            # set the current to the motors
            controller.set_stewart_platform_goal_current(
                np.round(current[1:], 0).astype(int).tolist()
            )
            viz.display(robot.state.q)

    except KeyboardInterrupt:
        pass

    print("Robot is stiff again.")
    controller.disable_torque()  # Enable torque
    controller.set_stewart_platform_operating_mode(
        3
    )  # Set operation mode to torque control
    controller.enable_torque()  # Enable torque for the Stewart platform motors


if __name__ == "__main__":
    main()

## gstreamer_client.py

"""Simple gstreamer webrtc consumer example."""

import argparse

import gi
from gst_signalling.utils import find_producer_peer_id_by_name

gi.require_version("Gst", "1.0")
from gi.repository import GLib, Gst  # noqa: E402


class GstConsumer:
    """Gstreamer webrtc consumer class."""

    def __init__(
        self,
        signalling_host: str,
        signalling_port: int,
        peer_name: str,
    ) -> None:
        """Initialize the consumer with signalling server details and peer name."""
        Gst.init(None)

        self.pipeline = Gst.Pipeline.new("webRTC-consumer")
        self.source = Gst.ElementFactory.make("webrtcsrc")

        if not self.pipeline:
            print("Pipeline could be created.")
            exit(-1)

        if not self.source:
            print(
                "webrtcsrc component could not be created. Please make sure that the plugin is installed \
                (see https://gitlab.freedesktop.org/gstreamer/gst-plugins-rs/-/tree/main/net/webrtc)"
            )
            exit(-1)

        self.pipeline.add(self.source)

        peer_id = find_producer_peer_id_by_name(
            signalling_host, signalling_port, peer_name
        )
        print(f"found peer id: {peer_id}")

        self.source.connect("pad-added", self.webrtcsrc_pad_added_cb)
        signaller = self.source.get_property("signaller")
        signaller.set_property("producer-peer-id", peer_id)
        signaller.set_property("uri", f"ws://{signalling_host}:{signalling_port}")

    def dump_latency(self) -> None:
        """Dump the current pipeline latency."""
        query = Gst.Query.new_latency()
        self.pipeline.query(query)
        print(f"Pipeline latency {query.parse_latency()}")

    def _configure_webrtcbin(self, webrtcsrc: Gst.Element) -> None:
        if isinstance(webrtcsrc, Gst.Bin):
            webrtcbin_name = "webrtcbin0"
            webrtcbin = webrtcsrc.get_by_name(webrtcbin_name)
            assert webrtcbin is not None
            # jitterbuffer has a default 200 ms buffer.
            webrtcbin.set_property("latency", 50)

    def webrtcsrc_pad_added_cb(self, webrtcsrc: Gst.Element, pad: Gst.Pad) -> None:
        """Add webrtcsrc elements when a new pad is added."""
        self._configure_webrtcbin(webrtcsrc)
        if pad.get_name().startswith("video"):  # type: ignore[union-attr]
            # webrtcsrc automatically decodes and convert the video
            sink = Gst.ElementFactory.make("fpsdisplaysink")
            assert sink is not None
            self.pipeline.add(sink)
            pad.link(sink.get_static_pad("sink"))  # type: ignore[arg-type]
            sink.sync_state_with_parent()

        elif pad.get_name().startswith("audio"):  # type: ignore[union-attr]
            # webrtcsrc automatically decodes and convert the audio
            sink = Gst.ElementFactory.make("autoaudiosink")
            assert sink is not None
            self.pipeline.add(sink)
            pad.link(sink.get_static_pad("sink"))  # type: ignore[arg-type]
            sink.sync_state_with_parent()

        GLib.timeout_add_seconds(5, self.dump_latency)

    def __del__(self) -> None:
        """Destructor to clean up GStreamer resources."""
        Gst.deinit()

    def get_bus(self) -> Gst.Bus:
        """Get the GStreamer bus for the pipeline."""
        return self.pipeline.get_bus()

    def play(self) -> None:
        """Start the GStreamer pipeline."""
        ret = self.pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            print("Error starting playback.")
            exit(-1)
        print("playing ... (ctrl+c to quit)")

    def stop(self) -> None:
        """Stop the GStreamer pipeline."""
        print("stopping")
        self.pipeline.send_event(Gst.Event.new_eos())
        self.pipeline.set_state(Gst.State.NULL)


def process_msg(bus: Gst.Bus, pipeline: Gst.Pipeline) -> bool:
    """Process messages from the GStreamer bus."""
    msg = bus.timed_pop_filtered(10 * Gst.MSECOND, Gst.MessageType.ANY)
    if msg:
        if msg.type == Gst.MessageType.ERROR:
            err, debug = msg.parse_error()
            print(f"Error: {err}, {debug}")
            return False
        elif msg.type == Gst.MessageType.EOS:
            print("End-Of-Stream reached.")
            return False
        elif msg.type == Gst.MessageType.LATENCY:
            if pipeline:
                try:
                    pipeline.recalculate_latency()
                except Exception as e:
                    print("failed to recalculate warning, exception: %s" % str(e))
        # else:
        #    print(f"Message: {msg.type}")
    return True


def main() -> None:
    """Run the main function."""
    parser = argparse.ArgumentParser(description="webrtc gstreamer simple consumer")
    parser.add_argument(
        "--signaling-host",
        default="127.0.0.1",
        help="Gstreamer signaling host - Reachy Mini ip",
    )
    parser.add_argument(
        "--signaling-port", default=8443, help="Gstreamer signaling port"
    )

    args = parser.parse_args()

    consumer = GstConsumer(
        args.signaling_host,
        args.signaling_port,
        "reachymini",
    )
    consumer.play()

    # Wait until error or EOS
    bus = consumer.get_bus()
    try:
        while True:
            if not process_msg(bus, consumer.pipeline):
                break

    except KeyboardInterrupt:
        print("User exit")
    finally:
        consumer.stop()


if __name__ == "__main__":
    main()

## joy_controller.py

#!/usr/bin/env python3
"""Control Reachy Mini's head yaw angle with a joystick.

This script connects to a Reachy Mini robot and allows you to pilot its head's
left-right rotation (yaw) using the horizontal axis of a connected joystick.

The yaw angle is mapped to a full range of +-pi/2 radians (+-90 degrees).
The value from the right joystick is also printed but is not used for control.

CONTROLS:
- LEFT JOYSTICK (Left/Right): Control head yaw angle.
- CIRCLE / B BUTTON (Button 1): Quit the application safely.
- CTRL-C: Quit the application.
"""

# Standard library imports
import os
import sys
import time

# Third-party imports
import numpy as np
import pygame

# Local application/library-specific imports
from reachy_mini import ReachyMini, utils

# --- Configuration ---
CONTROL_LOOP_RATE = 0.02
# Maximum yaw angle. The joystick's -1 to 1 input will be mapped to this range.
YAW_ANGLE_LIMIT = np.pi / 4 * 1.3  # Radians

# To use pygame "headlessly" (without a GUI window).
os.environ["SDL_VIDEODRIVER"] = "dummy"

# --- Controller Bindings Comment ---
# PS4 controller:
# Button 1 = O (Circle)
# Axis 0: Left Joy Left/Right   (-1 left, 1 right)
# Axis 3 or 4: Right Joy Left/Right
#
# XBOX controller:
# Button 1 = B
# Axis 0: Left Joy Left/Right
# Axis 2 or 3: Right Joy Left/Right


class Controller:
    """Handle joystick input using pygame."""

    def __init__(self, deadzone: float = 0.08):
        """Initialize the controller and find the first joystick.

        Args:
            deadzone (float): Axis value below which input is ignored.

        Raises:
            IOError: If no joystick is found.

        """
        pygame.init()
        pygame.joystick.init()

        if pygame.joystick.get_count() < 1:
            raise IOError("No joystick controller found.")

        self.joystick: pygame.joystick.Joystick = pygame.joystick.Joystick(0)
        self.deadzone = deadzone
        print(f"Initialized joystick: {self.joystick.get_name()}")

    def _apply_deadzone(self, value: float) -> float:
        """Apply a deadzone to a joystick axis value."""
        return value if abs(value) > self.deadzone else 0.0

    def get_horizontal_inputs(self) -> tuple[float, float]:
        """Read the horizontal axes of the left and right joysticks.

        Returns:
            tuple[float, float]: (left_joy_h, right_joy_h) from -1.0 to 1.0.

        """
        pygame.event.pump()  # Update pygame's internal event state.

        left_joy_h = self._apply_deadzone(self.joystick.get_axis(0))

        # Right joystick horizontal axis can be 2, 3 or 4 depending on controller
        right_joy_h = 0.0
        if self.joystick.get_numaxes() > 3:
            right_joy_h = self._apply_deadzone(self.joystick.get_axis(3))
        elif self.joystick.get_numaxes() > 2:
            right_joy_h = self._apply_deadzone(self.joystick.get_axis(2))

        return left_joy_h, right_joy_h

    def check_for_quit(self) -> bool:
        """Check pygame events for a quit signal.

        Returns:
            bool: True if the designated quit button (Circle/B) is pressed.

        """
        for event in pygame.event.get():
            if event.type == pygame.JOYBUTTONDOWN:
                if self.joystick.get_button(1):  # Button 1 is Circle/B
                    print("\nQuit button pressed.")
                    return True
        return False


def main() -> None:
    """Run the main joystick control loop."""
    try:
        controller = Controller()
    except IOError as e:
        print(f"Error: {e}", file=sys.stderr)
        return

    print("Connecting to Reachy Mini...")
    try:
        # The 'with' statement ensures the robot is properly handled on exit
        with ReachyMini(automatic_body_yaw=True) as mini:
            print("Robot connected.")
            # print("Robot connected. Waking up...")
            # mini.wake_up()

            print("\n" + "=" * 50)
            print("  Reachy Head Yaw Joystick Controller")
            print("  CONTROLS: [Left Stick] to turn | [Circle/B] to quit")
            print("=" * 50 + "\n")

            while True:
                if controller.check_for_quit():
                    break

                # Get scaled joystick values
                left_joy, right_joy = controller.get_horizontal_inputs()

                # Map joystick input (-1 to 1) to the desired angle range
                target_yaw = left_joy * YAW_ANGLE_LIMIT

                target_body_yaw = right_joy * YAW_ANGLE_LIMIT

                # Define the target pose: x,y,z and roll,pitch,yaw
                target_position = np.array([0, 0, 0.0])
                target_orientation = np.array([0, 0, target_yaw])

                # Create and send the command to the robot
                mini.set_target(
                    utils.create_head_pose(
                        *target_position, *target_orientation, degrees=False
                    ),
                    body_yaw=target_body_yaw,
                )

                # Print status, overwriting the line
                print(
                    f"\rSending Yaw: {target_yaw:6.2f} rad | "
                    f"Unused Right Joy: {right_joy:6.2f}",
                    end="",
                )
                sys.stdout.flush()

                time.sleep(CONTROL_LOOP_RATE)

    except KeyboardInterrupt:
        print("\nCTRL+C detected. Shutting down...")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}", file=sys.stderr)
    finally:
        print("\n\nApplication finished. Robot will go to sleep.")
        pygame.quit()


if __name__ == "__main__":
    main()

## measure_tracking.py

#!/usr/bin/env python3
"""ReachyMini dance-run tracker: generate + measure at 200 Hz, per-move plots.

What it does
------------
- Iterates over all AVAILABLE_MOVES.
- For each move, runs it for a fixed number of beats at a fixed BPM.
- Single client loop: sends targets and measures current pose at 200 Hz.
- Errors via distance_between_poses: translation [mm], angular [deg], combined [magic-mm].
- Saves per-move data and two figures in a run folder:
    measures/YYYY-MM-DD_HH-MM-SS/<move>.npz
    measures/YYYY-MM-DD_HH-MM-SS/<move>_errors.png
    measures/YYYY-MM-DD_HH-MM-SS/<move>_xyzrpy_vs_time.png
- Logs a warning on every sample where the target pose equals the previous one.

Dependencies: numpy, matplotlib, scipy, reachy_mini
Style: ruff-compatible docstrings and type hints.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from reachy_mini_dances_library.collection.dance import AVAILABLE_MOVES
from scipy.spatial.transform import Rotation as R

from reachy_mini import ReachyMini, utils
from reachy_mini.utils.interpolation import distance_between_poses

# ---------------- Configuration (tweak as needed) ----------------
BPM: float = 120.0  # tempo for all moves
BEATS_PER_MOVE: float = 30.0  # duration per move
SAMPLE_HZ: float = 200.0  # control + measurement rate
NEUTRAL_POS = np.array([0.0, 0.0, 0.0])  # meters
NEUTRAL_EUL = np.zeros(3)  # radians
# -----------------------------------------------------------------


def setup_logging() -> None:
    """Configure console logging."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )


def create_run_dir(base: Path = Path("measures")) -> Path:
    """Create and return a timestamped directory for this run."""
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out = base / stamp
    out.mkdir(parents=True, exist_ok=True)
    return out


def plot_errors_stack(
    t_abs: np.ndarray,
    trans_mm: np.ndarray,
    ang_deg: np.ndarray,
    magic_mm: np.ndarray,
    title_suffix: str,
    out_png: Path,
    beat_period_s: float | None = None,
) -> None:
    """Create a 3-row vertical stack with shared X axis and save as PNG."""
    if t_abs.size == 0:
        logging.warning("No samples to plot for %s", title_suffix)
        return
    t = t_abs - t_abs[0]
    fig, axes = plt.subplots(
        nrows=3, ncols=1, sharex=True, figsize=(12, 9), constrained_layout=True
    )

    ax = axes[0]
    ax.plot(t, trans_mm, linewidth=1.6, label="|Δx|")
    ax.set_ylabel("Position error [mm]")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    ax = axes[1]
    ax.plot(t, ang_deg, linewidth=1.6, label="|Δθ|")
    ax.set_ylabel("Angular error [deg]")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    ax = axes[2]
    ax.plot(t, magic_mm, linewidth=1.6, label="mm + deg")
    ax.set_ylabel("Combined error [magic-mm]")
    ax.set_xlabel("Time [s]")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    _draw_period_markers(axes, t, beat_period_s)

    fig.suptitle(f"Pose tracking errors vs time - {title_suffix}", fontsize=14)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_xyzrpy_stack(
    t_abs: np.ndarray,
    goal_pos_m: np.ndarray,
    present_pos_m: np.ndarray,
    goal_rpy_deg: np.ndarray,
    present_rpy_deg: np.ndarray,
    title_suffix: str,
    out_png: Path,
    beat_period_s: float | None = None,
) -> None:
    """Create a 6-row vertical stack (X/Y/Z in mm, Roll/Pitch/Yaw in deg), goal vs present."""
    if t_abs.size == 0:
        logging.warning("No samples to plot for %s", title_suffix)
        return
    t = t_abs - t_abs[0]

    # Convert positions to millimeters for plotting
    goal_pos_mm = goal_pos_m * 1000.0
    present_pos_mm = present_pos_m * 1000.0

    labels = [
        ("X position [mm]", 0),
        ("Y position [mm]", 1),
        ("Z position [mm]", 2),
        ("Roll [deg]", 0),
        ("Pitch [deg]", 1),
        ("Yaw [deg]", 2),
    ]

    fig, axes = plt.subplots(
        nrows=6, ncols=1, sharex=True, figsize=(12, 14), constrained_layout=True
    )

    # Positions (mm)
    for ax, (ylabel, idx) in zip(axes[:3], labels[:3]):
        ax.plot(t, goal_pos_mm[:, idx], linewidth=1.6, label=f"goal_{'xyz'[idx]}")
        ax.plot(t, present_pos_mm[:, idx], linewidth=1.6, label=f"present_{'xyz'[idx]}")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()

    # Orientations (deg)
    for ax, (ylabel, idx) in zip(axes[3:], labels[3:]):
        ax.plot(
            t,
            goal_rpy_deg[:, idx],
            linewidth=1.6,
            label=f"goal_{['roll', 'pitch', 'yaw'][idx]}",
        )
        ax.plot(
            t,
            present_rpy_deg[:, idx],
            linewidth=1.6,
            label=f"present_{['roll', 'pitch', 'yaw'][idx]}",
        )
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()

    axes[-1].set_xlabel("Time [s]")
    _draw_period_markers(axes, t, beat_period_s)
    fig.suptitle(f"Head position and orientation vs time - {title_suffix}", fontsize=14)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def _draw_period_markers(
    axes: np.ndarray, t: np.ndarray, beat_period_s: float | None
) -> None:
    if beat_period_s is None or beat_period_s <= 0.0 or t.size == 0:
        return
    duration = float(t[-1])
    if duration <= 0.0:
        return
    markers = np.arange(0.0, duration + 1e-9, beat_period_s)
    if markers.size == 0:
        markers = np.array([0.0])
    for ax in np.atleast_1d(axes):
        for marker in markers:
            ax.axvline(
                marker,
                color="tab:purple",
                linewidth=1.2,
                alpha=0.6,
                linestyle="--",
                zorder=3.0,
            )


def estimate_present_update_rate(
    t: np.ndarray, present_pos_m: np.ndarray, pos_tol_m: float = 1e-5
) -> float:
    """Estimate how often the present pose actually changes.

    We count samples where any XYZ component changes by more than pos_tol_m
    relative to the previous sample, then divide by total duration.

    Returns
    -------
    float
        Approximate update rate in Hz.

    """
    if t.size < 2:
        return 0.0
    diffs = np.abs(np.diff(present_pos_m, axis=0))
    changed = np.any(diffs > pos_tol_m, axis=1)
    n_changes = int(np.count_nonzero(changed))
    duration = float(t[-1] - t[0])
    return n_changes / duration if duration > 0 else 0.0


def save_npz(
    path: Path,
    data: Tuple[np.ndarray, ...],
    goal_pos_m: np.ndarray,
    present_pos_m: np.ndarray,
    goal_rpy_deg: np.ndarray,
    present_rpy_deg: np.ndarray,
) -> None:
    """Save measurements and extracted goal/present XYZ and RPY to .npz."""
    t, target, current, trans_mm, ang_deg, magic_mm = data
    np.savez_compressed(
        path,
        t=t,
        target=target,
        current=current,
        trans_mm=trans_mm,
        ang_deg=ang_deg,
        magic_mm=magic_mm,
        goal_pos_m=goal_pos_m,
        present_pos_m=present_pos_m,
        goal_rpy_deg=goal_rpy_deg,
        present_rpy_deg=present_rpy_deg,
    )


def run_one_move(
    mini: ReachyMini,
    move_name: str,
    move_def: Tuple,  # (move_fn, base_params, meta/desc)
    bpm: float,
    beats_total: float,
    sample_hz: float,
) -> Tuple[Tuple[np.ndarray, ...], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate targets for a single move and measure tracking until beats_total.

    Returns
    -------
    data
        Tuple (t, target_poses, current_poses, trans_mm, ang_deg, magic_mm)
    goal_pos_m
        Array (N, 3) from target_pose[:3, 3].
    present_pos_m
        Array (N, 3) from current_pose[:3, 3].
    goal_rpy_deg
        Array (N, 3) from R.from_matrix(target_pose[:3,:3]).as_euler("xyz", degrees=True).
    present_rpy_deg
        Array (N, 3) from R.from_matrix(current_pose[:3,:3]).as_euler("xyz", degrees=True).

    """
    period = 1.0 / sample_hz
    move_fn, base_params, _ = move_def

    # Params: copy and set a default waveform if present
    params: Dict = dict(base_params)
    if "waveform" in params:
        params["waveform"] = params.get("waveform", "sin")

    # Buffers
    t_list: list[float] = []
    target_list: list[np.ndarray] = []
    current_list: list[np.ndarray] = []
    trans_list: list[float] = []
    ang_list: list[float] = []
    magic_list: list[float] = []
    goal_pos_list: list[np.ndarray] = []
    present_pos_list: list[np.ndarray] = []
    goal_rpy_list: list[np.ndarray] = []
    present_rpy_list: list[np.ndarray] = []

    # Initialize
    current_pose = np.asarray(mini.get_current_head_pose(), dtype=float)
    last_target = current_pose.copy()

    # Beat-time and scheduler
    t_beats = 0.0
    prev_tick = time.perf_counter()
    next_sched = prev_tick

    logging.info(
        "Move '%s' start (BPM=%.1f, duration=%.1f beats)", move_name, bpm, beats_total
    )
    while t_beats < beats_total:
        next_sched += period

        # Offsets at current beat time
        offsets = move_fn(t_beats, **params)
        final_pos = NEUTRAL_POS + offsets.position_offset
        final_eul = NEUTRAL_EUL + offsets.orientation_offset
        final_ant = offsets.antennas_offset

        # Send target and read back current
        target_pose = utils.create_head_pose(*final_pos, *final_eul, degrees=False)
        mini.set_target(target_pose, antennas=final_ant)
        current_pose = np.asarray(mini.get_current_head_pose(), dtype=float)

        # Warning on unchanged target
        if np.array_equal(target_pose, last_target):
            logging.warning("Target pose unchanged for move '%s'.", move_name)
        last_target = target_pose

        # Errors
        d_trans_m, d_ang_rad, d_magic_mm = distance_between_poses(
            target_pose, current_pose
        )

        # Extract XYZ and RPY from both goal and present directly from the matrices
        goal_pos = target_pose[:3, 3].astype(float)
        present_pos = current_pose[:3, 3].astype(float)
        goal_rpy = (
            R.from_matrix(target_pose[:3, :3])
            .as_euler("xyz", degrees=True)
            .astype(float)
        )
        present_rpy = (
            R.from_matrix(current_pose[:3, :3])
            .as_euler("xyz", degrees=True)
            .astype(float)
        )

        # Append
        t_list.append(time.time())
        target_list.append(target_pose)
        current_list.append(current_pose)
        trans_list.append(float(d_trans_m * 1000.0))
        ang_list.append(float(np.degrees(d_ang_rad)))
        magic_list.append(float(d_magic_mm))
        goal_pos_list.append(goal_pos)
        present_pos_list.append(present_pos)
        goal_rpy_list.append(goal_rpy)
        present_rpy_list.append(present_rpy)

        # Timing and beat advance
        remaining = next_sched - time.perf_counter()
        if remaining > 0:
            time.sleep(remaining)
        now = time.perf_counter()
        dt_real = now - prev_tick
        prev_tick = now
        t_beats += dt_real * (bpm / 60.0)

    # Convert to arrays
    t = np.asarray(t_list, dtype=float)
    target_arr = np.asarray(target_list, dtype=float)
    current_arr = np.asarray(current_list, dtype=float)
    trans_arr = np.asarray(trans_list, dtype=float)
    ang_arr = np.asarray(ang_list, dtype=float)
    magic_arr = np.asarray(magic_list, dtype=float)
    goal_pos_m = np.asarray(goal_pos_list, dtype=float)
    present_pos_m = np.asarray(present_pos_list, dtype=float)
    goal_rpy_deg = np.asarray(goal_rpy_list, dtype=float)
    present_rpy_deg = np.asarray(present_rpy_list, dtype=float)

    return (
        (t, target_arr, current_arr, trans_arr, ang_arr, magic_arr),
        goal_pos_m,
        present_pos_m,
        goal_rpy_deg,
        present_rpy_deg,
    )


def main() -> None:
    """Run all AVAILABLE_MOVES sequentially, saving per-move data and plots."""
    setup_logging()
    run_dir = create_run_dir(Path("measures"))

    with ReachyMini() as mini:
        mini.wake_up()
        try:
            for move_name, move_def in AVAILABLE_MOVES.items():
                data, goal_pos_m, present_pos_m, goal_rpy_deg, present_rpy_deg = (
                    run_one_move(
                        mini=mini,
                        move_name=move_name,
                        move_def=move_def,
                        bpm=BPM,
                        beats_total=BEATS_PER_MOVE,
                        sample_hz=SAMPLE_HZ,
                    )
                )

                rate_hz = estimate_present_update_rate(
                    data[0], present_pos_m, pos_tol_m=1e-5
                )
                logging.info(
                    "Estimated present pose update rate for '%s': %.1f Hz",
                    move_name,
                    rate_hz,
                )

                # Save data and plots for this move
                npz_path = run_dir / f"{move_name}.npz"
                png_errors = run_dir / f"{move_name}_errors.png"
                png_xyzrpy = run_dir / f"{move_name}_xyzrpy_vs_time.png"

                save_npz(
                    npz_path,
                    data,
                    goal_pos_m,
                    present_pos_m,
                    goal_rpy_deg,
                    present_rpy_deg,
                )
                t, _target, _current, trans_mm, ang_deg, magic_mm = data
                beat_period_s = 60.0 / BPM if BPM > 0 else None

                plot_errors_stack(
                    t_abs=t,
                    trans_mm=trans_mm,
                    ang_deg=ang_deg,
                    magic_mm=magic_mm,
                    title_suffix=move_name,
                    out_png=png_errors,
                    beat_period_s=beat_period_s,
                )
                plot_xyzrpy_stack(
                    t_abs=t,
                    goal_pos_m=goal_pos_m,
                    present_pos_m=present_pos_m,
                    goal_rpy_deg=goal_rpy_deg,
                    present_rpy_deg=present_rpy_deg,
                    title_suffix=move_name,
                    out_png=png_xyzrpy,
                    beat_period_s=beat_period_s,
                )

                logging.info("Saved %s, %s and %s", npz_path, png_errors, png_xyzrpy)
        except KeyboardInterrupt:
            logging.info(
                "Interrupted by user. Finishing current move and saving what is available."
            )
        finally:
            mini.goto_sleep()
            logging.info("Run folder: %s", run_dir)


if __name__ == "__main__":
    main()

## mini_body_yaw_gui.py

"""Reachy Mini Head Position GUI Example."""

import time
import tkinter as tk

import numpy as np
from scipy.spatial.transform import Rotation as R

from reachy_mini import ReachyMini


def main():
    """Run a GUI to set the head position and orientation of Reachy Mini."""
    with ReachyMini() as mini:
        # with ReachyMini(automatic_body_yaw=False) as mini:
        t0 = time.time()

        root = tk.Tk()
        slider_length = 200
        root.title("Target Position and Orientation")

        roll_var = tk.DoubleVar(value=0.0)
        pitch_var = tk.DoubleVar(value=0.0)
        yaw_var = tk.DoubleVar(value=0.0)

        tk.Label(root, text="Roll (deg):").grid(row=0, column=0)
        tk.Scale(
            root,
            variable=roll_var,
            from_=-180,
            to=180,
            orient=tk.HORIZONTAL,
            length=slider_length,
        ).grid(row=0, column=1)
        tk.Label(root, text="Pitch (deg):").grid(row=1, column=0)
        tk.Scale(
            root,
            variable=pitch_var,
            from_=-180,
            to=180,
            orient=tk.HORIZONTAL,
            length=slider_length,
        ).grid(row=1, column=1)
        tk.Label(root, text="Yaw (deg):").grid(row=2, column=0)
        tk.Scale(
            root,
            variable=yaw_var,
            from_=-180,
            to=180,
            orient=tk.HORIZONTAL,
            length=slider_length,
        ).grid(row=2, column=1)

        # Add sliders for X, Y, Z position
        x_var = tk.DoubleVar(value=0.0)
        y_var = tk.DoubleVar(value=0.0)
        z_var = tk.DoubleVar(value=0.0)

        tk.Label(root, text="X (m):").grid(row=3, column=0)
        tk.Scale(
            root,
            variable=x_var,
            from_=-0.05,
            to=0.05,
            resolution=0.001,
            orient=tk.HORIZONTAL,
            length=slider_length,
        ).grid(row=3, column=1)
        tk.Label(root, text="Y (m):").grid(row=4, column=0)
        tk.Scale(
            root,
            variable=y_var,
            from_=-0.05,
            to=0.05,
            resolution=0.001,
            orient=tk.HORIZONTAL,
            length=slider_length,
        ).grid(row=4, column=1)
        tk.Label(root, text="Z (m):").grid(row=5, column=0)
        tk.Scale(
            root,
            variable=z_var,
            from_=-0.05,
            to=0.05,
            resolution=0.001,
            orient=tk.HORIZONTAL,
            length=slider_length,
        ).grid(row=5, column=1)

        # Add slider for Body Yaw
        body_yaw_var = tk.DoubleVar(value=0.0)
        tk.Label(root, text="Body Yaw (deg):").grid(row=6, column=0)
        tk.Scale(
            root,
            variable=body_yaw_var,
            from_=-180,
            to=180,
            orient=tk.HORIZONTAL,
            length=slider_length,
        ).grid(row=6, column=1)

        # Add checkbox for automatic body yaw
        # automatic_body_yaw_var = tk.BooleanVar(value=True)
        # tk.Checkbutton(
        #     root,
        #     text="Manual Body Yaw",
        #     variable=automatic_body_yaw_var,
        # ).grid(row=7, column=0)

        # Run the GUI in a non-blocking way
        root.update()

        while True:
            t = time.time() - t0
            target = np.deg2rad(30) * np.sin(2 * np.pi * 0.5 * t)

            head = np.eye(4)
            head[:3, 3] = [0, 0, 0.0]

            # Read values from the GUI
            roll = np.deg2rad(roll_var.get())
            pitch = np.deg2rad(pitch_var.get())
            yaw = np.deg2rad(yaw_var.get())
            head[:3, :3] = R.from_euler(
                "xyz", [roll, pitch, yaw], degrees=False
            ).as_matrix()
            head[:3, 3] = [x_var.get(), y_var.get(), z_var.get()]

            root.update()

            # mini.head_kinematics.automatic_body_yaw = not automatic_body_yaw_var.get()

            mini.set_target(
                head=head,
                antennas=np.array([target, -target]),
                body_yaw=np.deg2rad(body_yaw_var.get()),
            )
            time.sleep(0.02)


if __name__ == "__main__":
    main()

## sound_doa.py

"""Reachy Mini sound playback example.

Open a wav and push samples to the speaker. This is a toy example, in real
conditions output from a microphone or a text-to-speech engine would be
 pushed to the speaker instead.
"""

import logging
import time

import numpy as np

from reachy_mini import ReachyMini


def main() -> None:
    """Play a wav file by pushing samples to the audio device."""
    logging.basicConfig(
        level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    with ReachyMini(log_level="DEBUG", automatic_body_yaw=True) as mini:
        last_doa = -1
        THRESHOLD = 0.004  # ~2 degrees
        while True:
            doa = mini.media.audio.get_DoA()
            print(f"DOA: {doa}")
            if doa[1] and np.abs(doa[0] - last_doa) > THRESHOLD:
                print(f"  Speech detected at {doa[0]:.1f}°")
                p_head = [np.sin(doa[0]), np.cos(doa[0]), 0.0]
                print(
                    f"  Pointing to x={p_head[0]:.2f}, y={p_head[1]:.2f}, z={p_head[2]:.2f}"
                )
                T_world_head = mini.get_current_head_pose()
                R_world_head = T_world_head[:3, :3]
                p_world = R_world_head @ p_head
                print(
                    f"  In world coordinates: x={p_world[0]:.2f}, y={p_world[1]:.2f}, z={p_world[2]:.2f}"
                )
                mini.look_at_world(*p_world, duration=0.5)
                last_doa = doa[0]
            else:
                if not doa[1]:
                    print("  No speech detected")
                else:
                    print(
                        f"  Small change in DOA: {doa[0]:.1f}° (last was {last_doa:.1f}°). Not moving."
                    )
                time.sleep(0.5)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Exiting...")

## sound_play.py

"""Reachy Mini sound playback example.

Open a wav and push samples to the speaker. This is a toy example, in real
conditions output from a microphone or a text-to-speech engine would be
 pushed to the speaker instead.
"""

import argparse
import logging
import os
import time

import numpy as np
import scipy
import soundfile as sf

from reachy_mini import ReachyMini
from reachy_mini.utils.constants import ASSETS_ROOT_PATH

INPUT_FILE = os.path.join(ASSETS_ROOT_PATH, "wake_up.wav")


def main(backend: str) -> None:
    """Play a wav file by pushing samples to the audio device."""
    logging.basicConfig(
        level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    with ReachyMini(log_level="DEBUG", media_backend=backend) as mini:
        data, samplerate_in = sf.read(INPUT_FILE, dtype="float32")

        if samplerate_in != mini.media.get_audio_samplerate():
            data = scipy.signal.resample(
                data,
                int(len(data) * (mini.media.get_audio_samplerate() / samplerate_in)),
            )
        if data.ndim > 1:  # convert to mono
            data = np.mean(data, axis=1)

        mini.media.start_playing()
        print("Playing audio...")
        # Push samples in chunks
        chunk_size = 1024
        for i in range(0, len(data), chunk_size):
            chunk = data[i : i + chunk_size]
            mini.media.push_audio_sample(chunk)

        time.sleep(1)  # wait a bit to ensure all samples are played
        mini.media.stop_playing()
        print("Playback finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plays a wav file on Reachy Mini's speaker."
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["default", "gstreamer"],
        default="default",
        help="Media backend to use.",
    )

    args = parser.parse_args()
    main(backend=args.backend)

## sound_record.py

"""Reachy Mini sound recording example."""

import argparse
import logging
import time

import numpy as np
import soundfile as sf

from reachy_mini import ReachyMini

DURATION = 5  # seconds
OUTPUT_FILE = "recorded_audio.wav"


def main(backend: str) -> None:
    """Record audio for 5 seconds and save to a WAV file."""
    logging.basicConfig(
        level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    with ReachyMini(log_level="INFO", media_backend=backend) as mini:
        print(f"Recording for {DURATION} seconds...")
        audio_samples = []
        t0 = time.time()
        mini.media.start_recording()
        while time.time() - t0 < DURATION:
            sample = mini.media.get_audio_sample()

            if sample is not None:
                audio_samples.append(sample)
            else:
                print("No audio data available yet...")
            if backend == "default":
                time.sleep(0.2)
        mini.media.stop_recording()

        # Concatenate all samples and save
        if audio_samples:
            audio_data = np.concatenate(audio_samples, axis=0)
            samplerate = mini.media.get_audio_samplerate()
            sf.write(OUTPUT_FILE, audio_data, samplerate)
            print(f"Audio saved to {OUTPUT_FILE}")
        else:
            print("No audio data recorded.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Records audio from Reachy Mini's microphone."
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["default", "gstreamer"],
        default="default",
        help="Media backend to use.",
    )

    args = parser.parse_args()
    main(backend=args.backend)

## goto_interpolation_playground.py

"""Reachy Mini Goto Target Interpolation Playground.

This example demonstrates the different interpolation methods available in Reachy Mini
for moving the head and/or antennas to a target pose. It tests various methods such as linear,
minjerk, ease, and cartoon, allowing the user to observe how each method affects the
motion of the head and antennas.
"""

import numpy as np

from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose
from reachy_mini.utils.interpolation import InterpolationTechnique


def main():
    """Run the different interpolation methods."""
    with ReachyMini(media_backend="no_media") as mini:
        try:
            for method in InterpolationTechnique:
                print(f"Testing method: {method}")

                pose = create_head_pose(x=0, y=0, z=0, yaw=0)
                mini.goto_target(pose, duration=1.0, method=method)

                for _ in range(3):
                    pose = create_head_pose(
                        x=0.0, y=0.03, z=0, roll=5, yaw=-10, degrees=True
                    )
                    mini.goto_target(
                        pose,
                        antennas=np.deg2rad([-20, 20]),
                        duration=1.0,
                        method=method,
                    )

                    pose = create_head_pose(
                        x=0.0, y=-0.03, z=0, roll=-5, yaw=10, degrees=True
                    )
                    mini.goto_target(
                        pose,
                        antennas=np.deg2rad([20, -20]),
                        duration=1.0,
                        method=method,
                    )

                pose = create_head_pose(x=0, y=0, z=0, yaw=0)
                mini.goto_target(pose, duration=1.0, antennas=[0, 0], method=method)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()

## look_at_image.py

"""Demonstrate how to make Reachy Mini look at a point in an image.

When you click on the image, Reachy Mini will look at the point you clicked on.
It uses OpenCV to capture video from a camera and display it, and Reachy Mini's
look_at_image method to make the robot look at the specified point.

Note: The daemon must be running before executing this script.
"""

import argparse

import cv2

from reachy_mini import ReachyMini


def click(event, x, y, flags, param):
    """Handle mouse click events to get the coordinates of the click."""
    if event == cv2.EVENT_LBUTTONDOWN:
        param["just_clicked"] = True
        param["x"] = x
        param["y"] = y


def main(backend: str) -> None:
    """Show the camera feed from Reachy Mini and make it look at clicked points."""
    state = {"x": 0, "y": 0, "just_clicked": False}

    cv2.namedWindow("Reachy Mini Camera")
    cv2.setMouseCallback("Reachy Mini Camera", click, param=state)

    print("Click on the image to make ReachyMini look at that point.")
    print("Press 'q' to quit the camera feed.")
    with ReachyMini(media_backend=backend) as reachy_mini:
        try:
            while True:
                frame = reachy_mini.media.get_frame()

                if frame is None:
                    print("Failed to grab frame.")
                    continue

                cv2.imshow("Reachy Mini Camera", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Exiting...")
                    break

                if state["just_clicked"]:
                    reachy_mini.look_at_image(state["x"], state["y"], duration=0.3)
                    state["just_clicked"] = False
        except KeyboardInterrupt:
            print("Interrupted. Closing viewer...")
        finally:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Display Reachy Mini's camera feed and make it look at clicked points."
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["default", "gstreamer"],
        default="default",
        help="Media backend to use.",
    )

    args = parser.parse_args()
    main(backend=args.backend)

## mini_head_position_gui.py

"""Reachy Mini Head Position GUI Example."""

import time
import tkinter as tk

import numpy as np
from scipy.spatial.transform import Rotation as R

from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose


def main():
    """Run a GUI to set the head position and orientation of Reachy Mini."""
    with ReachyMini(media_backend="no_media") as mini:
        t0 = time.time()

        root = tk.Tk()
        root.title("Set Head Euler Angles")

        roll_var = tk.DoubleVar(value=0.0)
        pitch_var = tk.DoubleVar(value=0.0)
        yaw_var = tk.DoubleVar(value=0.0)

        tk.Label(root, text="Roll (deg):").grid(row=0, column=0)
        tk.Scale(
            root, variable=roll_var, from_=-45, to=45, orient=tk.HORIZONTAL, length=200
        ).grid(row=0, column=1)
        tk.Label(root, text="Pitch (deg):").grid(row=1, column=0)
        tk.Scale(
            root, variable=pitch_var, from_=-45, to=45, orient=tk.HORIZONTAL, length=200
        ).grid(row=1, column=1)
        tk.Label(root, text="Yaw (deg):").grid(row=2, column=0)
        tk.Scale(
            root, variable=yaw_var, from_=-175, to=175, orient=tk.HORIZONTAL, length=200
        ).grid(row=2, column=1)

        # Add sliders for X, Y, Z position
        x_var = tk.DoubleVar(value=0.0)
        y_var = tk.DoubleVar(value=0.0)
        z_var = tk.DoubleVar(value=0.0)

        tk.Label(root, text="X (m):").grid(row=3, column=0)
        tk.Scale(
            root,
            variable=x_var,
            from_=-0.05,
            to=0.05,
            resolution=0.001,
            orient=tk.HORIZONTAL,
            length=200,
        ).grid(row=3, column=1)
        tk.Label(root, text="Y (m):").grid(row=4, column=0)
        tk.Scale(
            root,
            variable=y_var,
            from_=-0.05,
            to=0.05,
            resolution=0.001,
            orient=tk.HORIZONTAL,
            length=200,
        ).grid(row=4, column=1)
        tk.Label(root, text="Z (m):").grid(row=5, column=0)
        tk.Scale(
            root,
            variable=z_var,
            from_=-0.05,
            to=0.03,
            resolution=0.001,
            orient=tk.HORIZONTAL,
            length=200,
        ).grid(row=5, column=1)

        tk.Label(root, text="Body Yaw (deg):").grid(row=6, column=0)
        body_yaw_var = tk.DoubleVar(value=0.0)
        tk.Scale(
            root,
            variable=body_yaw_var,
            from_=-180,
            to=180,
            resolution=1.0,
            orient=tk.HORIZONTAL,
            length=200,
        ).grid(row=6, column=1)

        # add a checkbox to enable/disable collision checking
        collision_check_var = tk.BooleanVar(value=False)
        tk.Checkbutton(root, text="Check Collision", variable=collision_check_var).grid(
            row=7, column=1
        )

        mini.goto_target(create_head_pose(), antennas=[0.0, 0.0], duration=1.0)

        # Run the GUI in a non-blocking way
        root.update()

        try:
            while True:
                t = time.time() - t0
                target = np.deg2rad(30) * np.sin(2 * np.pi * 0.5 * t)

                head = np.eye(4)
                head[:3, 3] = [0, 0, 0.0]

                # Read values from the GUI
                roll = np.deg2rad(roll_var.get())
                pitch = np.deg2rad(pitch_var.get())
                yaw = np.deg2rad(yaw_var.get())
                head[:3, :3] = R.from_euler(
                    "xyz", [roll, pitch, yaw], degrees=False
                ).as_matrix()
                head[:3, 3] = [x_var.get(), y_var.get(), z_var.get()]

                root.update()

                mini.set_target(
                    head=head,
                    body_yaw=np.deg2rad(body_yaw_var.get()),
                    antennas=np.array([target, -target]),
                )
        except KeyboardInterrupt:
            pass
        finally:
            root.destroy()


if __name__ == "__main__":
    main()

## minimal_demo.py

"""Minimal demo for Reachy Mini."""

import time

import numpy as np

from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose

with ReachyMini(media_backend="no_media") as mini:
    mini.goto_target(create_head_pose(), antennas=[0.0, 0.0], duration=1.0)
    try:
        while True:
            t = time.time()

            antennas_offset = np.deg2rad(20 * np.sin(2 * np.pi * 0.5 * t))
            pitch = np.deg2rad(10 * np.sin(2 * np.pi * 0.5 * t))

            head_pose = create_head_pose(
                roll=0.0,
                pitch=pitch,
                yaw=0.0,
                degrees=False,
                mm=False,
            )
            mini.set_target(head=head_pose, antennas=(antennas_offset, antennas_offset))
    except KeyboardInterrupt:
        pass

## reachy_compliant_demo.py

"""Reachy Mini Compliant Demo.

This demo turns the Reachy Mini into compliant mode and compensates for the gravity of the robot platform to prevent it from falling down.

You can now gently push the robot and it will follow your movements. And when you stop pushing it, it will stay in place.
This is useful for applications like human-robot interaction, where you want the robot to be compliant and follow the user's movements.
"""

import time

from reachy_mini import ReachyMini

print(
    "This demo currently only works with Placo as the kinematics engine. Start the daemon with:\nreachy-mini-daemon --kinematics-engine Placo"
)
with ReachyMini(media_backend="no_media") as mini:
    try:
        mini.enable_gravity_compensation()

        print("Reachy Mini is now compliant. Press Ctrl+C to exit.")
        while True:
            # do nothing, just keep the program running
            time.sleep(0.02)
    except KeyboardInterrupt:
        pass
    finally:
        mini.disable_gravity_compensation()
        print("Exiting... Reachy Mini is stiff again.")

## recorded_moves_example.py

"""Demonstrate and play all available moves from a dataset for Reachy Mini.

Run :

python3 recorded_moves_example.py -l [dance, emotions]
"""

import argparse

from reachy_mini import ReachyMini
from reachy_mini.motion.recorded_move import RecordedMove, RecordedMoves


def main(dataset_path: str) -> None:
    """Connect to Reachy and run the main demonstration loop."""
    recorded_moves = RecordedMoves(dataset_path)

    print("Connecting to Reachy Mini...")
    with ReachyMini(use_sim=False, media_backend="no_media") as reachy:
        print("Connection successful! Starting dance sequence...\n")
        try:
            while True:
                for move_name in recorded_moves.list_moves():
                    move: RecordedMove = recorded_moves.get(move_name)
                    print(f"Playing move: {move_name}: {move.description}\n")
                    # print(f"params: {move.move_params}")
                    reachy.play_move(move, initial_goto_duration=1.0)

        except KeyboardInterrupt:
            print("\n Sequence interrupted by user. Shutting down.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Demonstrate and play all available dance moves for Reachy Mini."
    )
    parser.add_argument(
        "-l", "--library", type=str, default="dance", choices=["dance", "emotions"]
    )
    args = parser.parse_args()

    dataset_path = (
        "pollen-robotics/reachy-mini-dances-library"
        if args.library == "dance"
        else "pollen-robotics/reachy-mini-emotions-library"
    )
    main(dataset_path)

## rerun_viewer.py

"""Reachy Mini sound playback example.

Open a wav and push samples to the speaker. This is a toy example, in real
conditions output from a microphone or a text-to-speech engine would be
 pushed to the speaker instead.

It requires the 'rerun-loader-urdf' package to be installed. It's not on PyPI,
so you need to install it from the GitHub repository: pip install git+https://github.com/rerun-io/rerun-loader-python-example-urdf.git
"""

import logging
import time

from reachy_mini import ReachyMini
from reachy_mini.utils.rerun import Rerun


def main():
    """Play a wav file by pushing samples to the audio device."""
    logging.basicConfig(
        level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    with ReachyMini(log_level="DEBUG") as mini:
        try:
            mini.enable_gravity_compensation()
            rerun = Rerun(mini)
            rerun.start()

            print("Reachy Mini is now compliant. Press Ctrl+C to exit.")
            while True:
                # do nothing, just keep the program running
                time.sleep(0.02)

        except KeyboardInterrupt:
            mini.disable_gravity_compensation()
            rerun.stop()
            print("Exiting... Reachy Mini is stiff again.")


if __name__ == "__main__":
    main()

## sequence.py

"""Reachy Mini Motion Sequence Example."""

import time

import numpy as np
from scipy.spatial.transform import Rotation as R

from reachy_mini import ReachyMini

with ReachyMini(media_backend="no_media") as reachy_mini:
    reachy_mini.goto_target(np.eye(4), antennas=[0.0, 0.0], duration=1.0)
    try:
        while True:
            pose = np.eye(4)

            t = 0
            t0 = time.time()
            s = time.time()
            while time.time() - s < 2.0:
                t = time.time() - t0
                euler_rot = np.array([0, 0.0, 0.7 * np.sin(2 * np.pi * 0.5 * t)])
                rot_mat = R.from_euler("xyz", euler_rot, degrees=False).as_matrix()
                pose[:3, :3] = rot_mat
                reachy_mini.set_target(head=pose, antennas=[0, 0])
                time.sleep(0.01)

            s = time.time()
            while time.time() - s < 2.0:
                t = time.time() - t0
                euler_rot = np.array([0, 0.3 * np.sin(2 * np.pi * 0.5 * t), 0])
                rot_mat = R.from_euler("xyz", euler_rot, degrees=False).as_matrix()
                pose[:3, :3] = rot_mat
                reachy_mini.set_target(head=pose, antennas=[0, 0])
                time.sleep(0.01)

            s = time.time()
            while time.time() - s < 2.0:
                t = time.time() - t0
                euler_rot = np.array([0.3 * np.sin(2 * np.pi * 0.5 * t), 0, 0])
                rot_mat = R.from_euler("xyz", euler_rot, degrees=False).as_matrix()
                pose[:3, :3] = rot_mat
                reachy_mini.set_target(head=pose, antennas=[0, 0])
                time.sleep(0.01)

            s = time.time()
            while time.time() - s < 2.0:
                t = time.time() - t0
                pose = np.eye(4)
                pose[:3, 3][2] += 0.025 * np.sin(2 * np.pi * 0.5 * t)
                reachy_mini.set_target(head=pose, antennas=[0, 0])
                time.sleep(0.01)

            s = time.time()
            while time.time() - s < 2.0:
                t = time.time() - t0
                antennas = [
                    0.5 * np.sin(2 * np.pi * 0.5 * t),
                    -0.5 * np.sin(2 * np.pi * 0.5 * t),
                ]
                reachy_mini.set_target(head=pose, antennas=antennas)
                time.sleep(0.01)

            s = time.time()
            while time.time() - s < 5.0:
                t = time.time() - t0
                pose[:3, 3] = [
                    0.015 * np.sin(2 * np.pi * 1.0 * t),
                    0.015 * np.sin(2 * np.pi * 1.0 * t + np.pi / 2),
                    0.0,
                ]
                reachy_mini.set_target(head=pose, antennas=[0, 0])
                time.sleep(0.01)

            pose[:3, 3] = [0, 0, 0.0]
            reachy_mini.set_target(head=pose, antennas=[0, 0])

            time.sleep(0.5)

            pose[:3, 3] = [0.02, 0.02, 0.0]
            reachy_mini.set_target(head=pose, antennas=[0, 0])
            time.sleep(0.5)

            pose[:3, 3] = [0.00, 0.02, 0.0]
            euler_rot = np.array([0, 0, 0.5])
            rot_mat = R.from_euler("xyz", euler_rot, degrees=False).as_matrix()
            pose[:3, :3] = rot_mat
            reachy_mini.set_target(head=pose, antennas=[0, 0])
            time.sleep(0.5)

            pose[:3, 3] = [0.00, -0.02, 0.0]
            euler_rot = np.array([0, 0, -0.5])
            rot_mat = R.from_euler("xyz", euler_rot, degrees=False).as_matrix()
            pose[:3, :3] = rot_mat
            reachy_mini.set_target(head=pose, antennas=[0, 0])
            time.sleep(0.5)

            pose[:3, 3] = [0, 0, 0.0]
            reachy_mini.set_target(head=pose, antennas=[0, 0])
            time.sleep(2)

    except KeyboardInterrupt:
        pass
