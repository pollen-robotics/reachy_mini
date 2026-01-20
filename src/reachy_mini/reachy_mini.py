"""Reachy Mini class for controlling a simulated or real Reachy Mini robot.

This class provides methods to control the head and antennas of the Reachy Mini robot,
set their target positions, and perform various behaviors such as waking up and going to sleep.

It also includes methods for multimedia interactions like playing sounds and looking at specific points in the image frame or world coordinates.
"""

import asyncio
import json
import logging
import platform
import time
from typing import Dict, List, Literal, Optional, Union, cast

import cv2
import numpy as np
import numpy.typing as npt
import zenoh
from asgiref.sync import async_to_sync
from scipy.spatial.transform import Rotation as R

from reachy_mini.daemon.utils import daemon_check, is_local_camera_available
from reachy_mini.io.protocol import GotoTaskRequest
from reachy_mini.io.zenoh_client import ZenohClient
from reachy_mini.media.media_manager import MediaBackend, MediaManager
from reachy_mini.motion.move import Move
from reachy_mini.utils.interpolation import InterpolationTechnique, minimum_jerk

# Behavior definitions
INIT_HEAD_POSE = np.eye(4)

SLEEP_HEAD_JOINT_POSITIONS = [
    0,
    -0.9848156658225817,
    1.2624661884298831,
    -0.24390294527381684,
    0.20555342557667577,
    -1.2363885150358267,
    1.0032234352772091,
]


SLEEP_ANTENNAS_JOINT_POSITIONS = [-3.05, 3.05]
SLEEP_HEAD_POSE = np.array(
    [
        [0.911, 0.004, 0.413, -0.021],
        [-0.004, 1.0, -0.001, 0.001],
        [-0.413, -0.001, 0.911, -0.044],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

ConnectionMode = Literal["auto", "localhost_only", "network"]


class ReachyMini:
    """Reachy Mini class for controlling a simulated or real Reachy Mini robot.

    Args:
        connection_mode: Select how to connect to the daemon. Use
            `"localhost_only"` to restrict connections to daemons running on
            localhost, `"network"` to scout for daemons on the LAN, or `"auto"`
            (default) to try localhost first then fall back to the network.
        spawn_daemon (bool): If True, will spawn a daemon to control the robot, defaults to False.
        use_sim (bool): If True and spawn_daemon is True, will spawn a simulated robot, defaults to True.

    """

    def __init__(
        self,
        robot_name: str = "reachy_mini",
        connection_mode: ConnectionMode = "auto",
        spawn_daemon: bool = False,
        use_sim: bool = False,
        timeout: float = 5.0,
        automatic_body_yaw: bool = True,
        log_level: str = "INFO",
        media_backend: str = "default",
        localhost_only: Optional[bool] = None,
    ) -> None:
        """Initialize the Reachy Mini robot.

        Args:
            robot_name (str): Name of the robot, defaults to "reachy_mini".
            connection_mode: `"auto"` (default), `"localhost_only"` or `"network"`.
                `"auto"` will first try daemons on localhost and fall back to
                network discovery if no local daemon responds.
            localhost_only (Optional[bool]): Deprecated alias for the connection
                mode. Set `False` to search for network daemons. Will be removed
                in a future release.
            spawn_daemon (bool): If True, will spawn a daemon to control the robot, defaults to False.
            use_sim (bool): If True and spawn_daemon is True, will spawn a simulated robot, defaults to True.
            timeout (float): Timeout for the client connection, defaults to 5.0 seconds.
            automatic_body_yaw (bool): If True, the body yaw will be used to compute the IK and FK. Default is False.
            log_level (str): Logging level, defaults to "INFO".
            media_backend (str): Use "no_media" to disable media entirely. Any other value
                triggers auto-detection: Lite uses OpenCV, Wireless uses GStreamer (local)
                or WebRTC (remote) based on environment.

        It will try to connect to the daemon, and if it fails, it will raise an exception.

        """
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        self.robot_name = robot_name
        daemon_check(spawn_daemon, use_sim)
        normalized_mode = self._normalize_connection_mode(
            connection_mode, localhost_only
        )
        self.client, self.connection_mode = self._initialize_client(
            normalized_mode, timeout
        )
        self.set_automatic_body_yaw(automatic_body_yaw)
        self._last_head_pose: Optional[npt.NDArray[np.float64]] = None
        self.is_recording = False

        self.T_head_cam = np.eye(4)
        self.T_head_cam[:3, 3][:] = [0.0437, 0, 0.0512]
        self.T_head_cam[:3, :3] = np.array(
            [
                [0, 0, 1],
                [-1, 0, 0],
                [0, -1, 0],
            ]
        )

        self.media_manager = self._configure_mediamanager(media_backend, log_level)

    def __del__(self) -> None:
        """Destroy the Reachy Mini instance.

        The client is disconnected explicitly to avoid a thread pending issue.

        """
        if hasattr(self, "client"):
            self.client.disconnect()

    def __enter__(self) -> "ReachyMini":
        """Context manager entry point for Reachy Mini."""
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:  # type: ignore [no-untyped-def]
        """Context manager exit point for Reachy Mini."""
        self.media_manager.close()
        self.client.disconnect()

    @property
    def media(self) -> MediaManager:
        """Expose the MediaManager instance used by ReachyMini."""
        return self.media_manager

    @property
    def imu(self) -> Dict[str, List[float] | float] | None:
        """Get the current IMU data from the backend.

        Returns:
            dict with the following keys, or None if IMU is not available (Lite version)
            or no data received yet:
            - 'accelerometer': [x, y, z] in m/s^2
            - 'gyroscope': [x, y, z] in rad/s
            - 'quaternion': [w, x, y, z] orientation quaternion
            - 'temperature': float in °C

        Note:
            - Data is cached from the last Zenoh update at 50Hz
            - Quaternion is in [w, x, y, z] format

        Example:
            >>> imu_data = reachy.imu
            >>> if imu_data is not None:
            >>>     accel_x, accel_y, accel_z = imu_data['accelerometer']
            >>>     gyro_x, gyro_y, gyro_z = imu_data['gyroscope']
            >>>     quat_w, quat_x, quat_y, quat_z = imu_data['quaternion']
            >>>     temp = imu_data['temperature']

        """
        return self.client.get_current_imu_data()

    def _configure_mediamanager(
        self, media_backend: str, log_level: str
    ) -> MediaManager:
        daemon_status = self.client.get_status()
        is_wireless = daemon_status.get("wireless_version", False)

        # If no_media is requested, skip all media initialization
        if media_backend.lower() == "no_media":
            self.logger.info("No media backend requested.")
            mbackend = MediaBackend.NO_MEDIA
        else:
            if is_wireless:
                if is_local_camera_available():
                    # Local client on CM4: use GStreamer to read from unix socket
                    # This avoids WebRTC encode/decode overhead
                    if "no_video" in media_backend.lower():
                        mbackend = MediaBackend.GSTREAMER_NO_VIDEO
                        self.logger.info(
                            "Auto-detected: Wireless + local camera socket. "
                            "Using GStreamer audio-only backend (no WebRTC overhead)."
                        )
                    else:
                        mbackend = MediaBackend.GSTREAMER
                        self.logger.info(
                            "Auto-detected: Wireless + local camera socket. "
                            "Using GStreamer backend (no WebRTC overhead)."
                        )
                else:
                    # Remote client: use WebRTC for streaming
                    self.logger.info(
                        "Auto-detected: Wireless + remote client. "
                        "Using WebRTC backend for streaming."
                    )
                    mbackend = MediaBackend.WEBRTC
            else:
                # Lite version: use specified backend if compatible
                try:
                    mbackend = MediaBackend(media_backend.lower())
                    if mbackend == MediaBackend.WEBRTC:
                        self.logger.warning(
                            f"Incompatible media backend on Lite: {media_backend}, using default backend."
                        )
                        mbackend = MediaBackend.DEFAULT
                    # TODO : Remove when wheel is released !
                    elif "gstreamer" in media_backend.lower() and (
                        platform.system() == "Darwin" or platform.system() == "Windows"
                    ):
                        self.logger.warning(
                            f"Unsupported media backend on Lite for {platform.system()}: {media_backend}, using default backend."
                        )
                        mbackend = (
                            MediaBackend.DEFAULT_NO_VIDEO
                            if "no_video" in media_backend.lower()
                            else MediaBackend.DEFAULT
                        )
                    else:
                        self.logger.info(
                            f"Auto-detected: Lite. Using {mbackend} backend."
                        )
                except ValueError:
                    self.logger.warning(
                        f"Invalid media backend on Lite: {media_backend}, using default backend."
                    )
                    mbackend = MediaBackend.DEFAULT

        return MediaManager(
            use_sim=self.client.get_status()["simulation_enabled"],
            backend=mbackend,
            log_level=log_level,
            signalling_host=self.client.get_status()["wlan_ip"],
        )

    def _normalize_connection_mode(
        self,
        connection_mode: ConnectionMode,
        legacy_localhost_only: Optional[bool],
    ) -> ConnectionMode:
        """Normalize connection mode input, optionally honoring the legacy alias."""
        normalized = connection_mode.lower()
        if normalized not in {"auto", "localhost_only", "network"}:
            raise ValueError(
                "Invalid connection_mode. Use 'auto', 'localhost_only', or 'network'."
            )
        resolved = cast(ConnectionMode, normalized)

        if legacy_localhost_only is None:
            return resolved

        self.logger.warning(
            "The 'localhost_only' argument is deprecated and will be removed in a "
            "future release. Please switch to connection_mode."
        )

        if resolved != "auto":
            self.logger.warning(
                "Both connection_mode=%s and localhost_only=%s were provided. "
                "connection_mode takes precedence.",
                resolved,
                legacy_localhost_only,
            )
            return resolved

        return "localhost_only" if legacy_localhost_only else "network"

    def _initialize_client(
        self, requested_mode: ConnectionMode, timeout: float
    ) -> tuple[ZenohClient, ConnectionMode]:
        """Create a client according to the requested mode, adding auto fallback."""
        requested_mode = cast(ConnectionMode, requested_mode.lower())
        if requested_mode == "auto":
            try:
                client = self._connect_single(localhost_only=True, timeout=timeout)
                selected: ConnectionMode = "localhost_only"
            except Exception as err:
                self.logger.info(
                    "Auto connection: localhost attempt failed (%s). "
                    "Trying network discovery.",
                    err,
                )
                try:
                    client = self._connect_single(localhost_only=False, timeout=timeout)
                except (zenoh.ZError, TimeoutError):
                    raise ConnectionError(
                        "Auto connection: both localhost and network attempts failed. "
                        "Make sure a Reachy Mini daemon is running and accessible."
                    )

                selected = "network"
            self.logger.info("Connection mode selected: %s", selected)
            return client, selected

        if requested_mode == "localhost_only":
            try:
                client = self._connect_single(localhost_only=True, timeout=timeout)
            except (zenoh.ZError, TimeoutError):
                raise ConnectionError(
                    "Could not connect to daemon on localhost. Is the Reachy Mini daemon running?"
                )
            selected = "localhost_only"
        else:
            try:
                client = self._connect_single(localhost_only=False, timeout=timeout)
            except (zenoh.ZError, TimeoutError):
                raise ConnectionError(
                    "Network connection attempt failed. "
                    "Make sure a Reachy Mini daemon is running and accessible."
                )
            selected = "network"

        self.logger.info("Connection mode selected: %s", selected)
        return client, selected

    def _connect_single(self, localhost_only: bool, timeout: float) -> ZenohClient:
        """Connect once with the requested tunneling mode and guard cleanup."""
        client = ZenohClient(self.robot_name, localhost_only)
        client.wait_for_connection(timeout=timeout)
        return client

    def set_target(
        self,
        head: Optional[npt.NDArray[np.float64]] = None,  # 4x4 pose matrix
        antennas: Optional[
            Union[npt.NDArray[np.float64], List[float]]
        ] = None,  # [right_angle, left_angle] (in rads)
        body_yaw: Optional[float] = None,  # Body yaw angle in radians
    ) -> None:
        """Set the target pose of the head and/or the target position of the antennas.

        Args:
            head (Optional[np.ndarray]): 4x4 pose matrix representing the head pose.
            antennas (Optional[Union[np.ndarray, List[float]]]): 1D array with two elements representing the angles of the antennas in radians.
            body_yaw (Optional[float]): Body yaw angle in radians.

        Raises:
            ValueError: If neither head nor antennas are provided, or if the shape of head is not (4, 4), or if antennas is not a 1D array with two elements.

        """
        if head is None and antennas is None and body_yaw is None:
            raise ValueError(
                "At least one of head, antennas or body_yaw must be provided."
            )

        if head is not None and not head.shape == (4, 4):
            raise ValueError(f"Head pose must be a 4x4 matrix, got shape {head.shape}.")

        if antennas is not None and not len(antennas) == 2:
            raise ValueError(
                "Antennas must be a list or 1D np array with two elements."
            )

        if body_yaw is not None and not isinstance(body_yaw, (int, float)):
            raise ValueError("body_yaw must be a float.")

        if head is not None:
            self.set_target_head_pose(head)

        if antennas is not None:
            self.set_target_antenna_joint_positions(list(antennas))
            # self._set_joint_positions(
            #     antennas_joint_positions=list(antennas),
            # )

        if body_yaw is not None:
            self.set_target_body_yaw(body_yaw)

        self._last_head_pose = head

        record: Dict[str, float | List[float] | List[List[float]]] = {
            "time": time.time(),
            "body_yaw": body_yaw if body_yaw is not None else 0.0,
        }
        if head is not None:
            record["head"] = head.tolist()
        if antennas is not None:
            record["antennas"] = list(antennas)
        if body_yaw is not None:
            record["body_yaw"] = body_yaw
        self._set_record_data(record)

    def goto_target(
        self,
        head: Optional[npt.NDArray[np.float64]] = None,  # 4x4 pose matrix
        antennas: Optional[
            Union[npt.NDArray[np.float64], List[float]]
        ] = None,  # [right_angle, left_angle] (in rads)
        duration: float = 0.5,  # Duration in seconds for the movement, default is 0.5 seconds.
        method: InterpolationTechnique = InterpolationTechnique.MIN_JERK,  # can be "linear", "minjerk", "ease" or "cartoon", default is "minjerk")
        body_yaw: float | None = 0.0,  # Body yaw angle in radians
    ) -> None:
        """Go to a target head pose and/or antennas position using task space interpolation, in "duration" seconds.

        Args:
            head (Optional[np.ndarray]): 4x4 pose matrix representing the target head pose.
            antennas (Optional[Union[np.ndarray, List[float]]]): 1D array with two elements representing the angles of the antennas in radians.
            duration (float): Duration of the movement in seconds.
            method (InterpolationTechnique): Interpolation method to use ("linear", "minjerk", "ease", "cartoon"). Default is "minjerk".
            body_yaw (float | None): Body yaw angle in radians. Use None to keep the current yaw.

        Raises:
            ValueError: If neither head nor antennas are provided, or if duration is not positive.

        """
        if head is None and antennas is None and body_yaw is None:
            raise ValueError(
                "At least one of head, antennas or body_yaw must be provided."
            )

        if duration <= 0.0:
            raise ValueError(
                "Duration must be positive and non-zero. Use set_target() for immediate position setting."
            )

        req = GotoTaskRequest(
            head=(
                np.array(head, dtype=np.float64).flatten().tolist()
                if head is not None
                else None
            ),
            antennas=(
                np.array(antennas, dtype=np.float64).flatten().tolist()
                if antennas is not None
                else None
            ),
            duration=duration,
            method=method,
            body_yaw=body_yaw,
        )

        task_uid = self.client.send_task_request(req)
        self.client.wait_for_task_completion(task_uid, timeout=duration + 1.0)

    def wake_up(self) -> None:
        """Wake up the robot - go to the initial head position and play the wake up emote and sound."""
        self.goto_target(INIT_HEAD_POSE, antennas=[0.0, 0.0], duration=2)
        time.sleep(0.1)

        # Toudoum
        self.media.play_sound("wake_up.wav")

        # Roll 20° to the left
        pose = INIT_HEAD_POSE.copy()
        pose[:3, :3] = R.from_euler("xyz", [20, 0, 0], degrees=True).as_matrix()
        self.goto_target(pose, duration=0.2)

        # Go back to the initial position
        self.goto_target(INIT_HEAD_POSE, duration=0.2)

    def goto_sleep(self) -> None:
        """Put the robot to sleep by moving the head and antennas to a predefined sleep position."""
        # Check if we are too far from the initial position
        # Move to the initial position if necessary
        current_positions, _ = self.get_current_joint_positions()
        # init_positions = self.head_kinematics.ik(INIT_HEAD_POSE)
        # Todo : get init position from the daemon?
        init_positions = [
            6.959852054044218e-07,
            0.5251518455536499,
            -0.668710345667336,
            0.6067086443974802,
            -0.606711497194891,
            0.6687148024583701,
            -0.5251586523105128,
        ]
        dist = np.linalg.norm(np.array(current_positions) - np.array(init_positions))
        if dist > 0.2:
            self.goto_target(INIT_HEAD_POSE, antennas=[0.0, 0.0], duration=1)
            time.sleep(0.2)

        # Pfiou
        self.media.play_sound("go_sleep.wav")

        # # Move to the sleep position
        self.goto_target(
            SLEEP_HEAD_POSE, antennas=SLEEP_ANTENNAS_JOINT_POSITIONS, duration=2
        )

        self._last_head_pose = SLEEP_HEAD_POSE
        time.sleep(2)

    def look_at_image(
        self, u: int, v: int, duration: float = 1.0, perform_movement: bool = True
    ) -> npt.NDArray[np.float64]:
        """Make the robot head look at a point defined by a pixel position (u,v).

        # TODO image of reachy mini coordinate system

        Args:
            u (int): Horizontal coordinate in image frame.
            v (int): Vertical coordinate in image frame.
            duration (float): Duration of the movement in seconds. If 0, the head will snap to the position immediately.
            perform_movement (bool): If True, perform the movement. If False, only calculate and return the pose.

        Returns:
            np.ndarray: The calculated head pose as a 4x4 matrix.

        Raises:
            ValueError: If duration is negative.

        """
        if self.media_manager.camera is None:
            raise RuntimeError("Camera is not initialized.")

        # TODO this is false for the raspicam for now
        assert 0 < u < self.media_manager.camera.resolution[0], (
            f"u must be in [0, {self.media_manager.camera.resolution[0]}], got {u}."
        )
        assert 0 < v < self.media_manager.camera.resolution[1], (
            f"v must be in [0, {self.media_manager.camera.resolution[1]}], got {v}."
        )

        if duration < 0:
            raise ValueError("Duration can't be negative.")

        if self.media.camera is None or self.media.camera.camera_specs is None:
            raise RuntimeError("Camera specs not set.")

        points = np.array([[[u, v]]], dtype=np.float32)
        x_n, y_n = cv2.undistortPoints(
            points,
            self.media.camera.K,  # type: ignore
            self.media.camera.D,
        )[0, 0]

        ray_cam = np.array([x_n, y_n, 1.0])
        ray_cam /= np.linalg.norm(ray_cam)

        T_world_head = self.get_current_head_pose()
        T_world_cam = T_world_head @ self.T_head_cam

        R_wc = T_world_cam[:3, :3]
        t_wc = T_world_cam[:3, 3]

        ray_world = R_wc @ ray_cam

        P_world = t_wc + ray_world

        return self.look_at_world(
            x=P_world[0],
            y=P_world[1],
            z=P_world[2],
            duration=duration,
            perform_movement=perform_movement,
        )

    def look_at_world(
        self,
        x: float,
        y: float,
        z: float,
        duration: float = 1.0,
        perform_movement: bool = True,
    ) -> npt.NDArray[np.float64]:
        """Look at a specific point in 3D space in Reachy Mini's reference frame.

        TODO include image of reachy mini coordinate system

        Args:
            x (float): X coordinate in meters.
            y (float): Y coordinate in meters.
            z (float): Z coordinate in meters.
            duration (float): Duration of the movement in seconds. If 0, the head will snap to the position immediately.
            perform_movement (bool): If True, perform the movement. If False, only calculate and return the pose.

        Returns:
            np.ndarray: The calculated head pose as a 4x4 matrix.

        Raises:
            ValueError: If duration is negative.

        """
        if duration < 0:
            raise ValueError("Duration can't be negative.")

        # Head is at the origin, so vector from head to target position is directly the target position
        # TODO FIX : Actually, the head frame is not the origin frame wrt the kinematics. Close enough for now.
        target_position = np.array([x, y, z])
        target_vector = target_position / np.linalg.norm(
            target_position
        )  # normalize the vector

        # head_pointing straight vector
        straight_head_vector = np.array([1, 0, 0])

        # Calculate the rotation needed to align the head with the target vector
        v1 = straight_head_vector
        v2 = target_vector
        axis = np.cross(v1, v2)
        axis_norm = np.linalg.norm(axis)
        if axis_norm < 1e-8:
            # Vectors are (almost) parallel
            if np.dot(v1, v2) > 0:
                rot_mat = np.eye(3)
            else:
                # Opposite direction: rotate 180° around any perpendicular axis
                perp = np.array([0, 1, 0]) if abs(v1[0]) < 0.9 else np.array([0, 0, 1])
                axis = np.cross(v1, perp)
                axis /= np.linalg.norm(axis)
                rot_mat = R.from_rotvec(np.pi * axis).as_matrix()
        else:
            axis = axis / axis_norm
            angle = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))
            rotation_vector = angle * axis
            rot_mat = R.from_rotvec(rotation_vector).as_matrix()

        target_head_pose = np.eye(4)
        target_head_pose[:3, :3] = rot_mat

        # If perform_movement is True, execute the movement
        if perform_movement:
            # If duration is specified, use the goto_target method to move smoothly
            # Otherwise, set the position immediately
            if duration > 0:
                self.goto_target(target_head_pose, duration=duration)
            else:
                self.set_target(target_head_pose)

        return target_head_pose

    def _goto_joint_positions(
        self,
        head_joint_positions: Optional[
            List[float]
        ] = None,  # [yaw, stewart_platform x 6] length 7
        antennas_joint_positions: Optional[
            List[float]
        ] = None,  # [right_angle, left_angle] length 2
        duration: float = 0.5,  # Duration in seconds for the movement
    ) -> None:
        """Go to a target head joint positions and/or antennas joint positions using joint space interpolation, in "duration" seconds.

        [Internal] Go to a target head joint positions and/or antennas joint positions using joint space interpolation, in "duration" seconds.

        Args:
            head_joint_positions (Optional[List[float]]): List of head joint positions in radians (length 7).
            antennas_joint_positions (Optional[List[float]]): List of antennas joint positions in radians (length 2).
            duration (float): Duration of the movement in seconds. Default is 0.5 seconds.

        Raises:
            ValueError: If neither head_joint_positions nor antennas_joint_positions are provided, or if duration is not positive.

        """
        if duration <= 0.0:
            raise ValueError(
                "Duration must be positive and non-zero. Use set_target() for immediate position setting."
            )

        cur_head, cur_antennas = self.get_current_joint_positions()
        current = cur_head + cur_antennas

        target = []
        if head_joint_positions is not None:
            target.extend(head_joint_positions)
        else:
            target.extend(cur_head)
        if antennas_joint_positions is not None:
            target.extend(antennas_joint_positions)
        else:
            target.extend(cur_antennas)

        traj = minimum_jerk(np.array(current), np.array(target), duration)

        t0 = time.time()
        while time.time() - t0 < duration:
            t = time.time() - t0
            angles = traj(t)

            head_joint = angles[:7]  # First 7 angles for the head
            antennas_joint = angles[7:]

            self._set_joint_positions(list(head_joint), list(antennas_joint))
            time.sleep(0.01)

    def get_current_joint_positions(self) -> tuple[list[float], list[float]]:
        """Get the current joint positions of the head and antennas.

        Get the current joint positions of the head and antennas (in rad)

        Returns:
            tuple: A tuple containing two lists:
                - List of head joint positions (rad) (length 7).
                - List of antennas joint positions (rad) (length 2).

        """
        return self.client.get_current_joints()

    def get_present_antenna_joint_positions(self) -> list[float]:
        """Get the present joint positions of the antennas.

        Get the present joint positions of the antennas (in rad)

        Returns:
            list: A list of antennas joint positions (rad) (length 2).

        """
        return self.get_current_joint_positions()[1]

    def get_current_head_pose(self) -> npt.NDArray[np.float64]:
        """Get the current head pose as a 4x4 matrix.

        Get the current head pose as a 4x4 matrix.

        Returns:
            np.ndarray: A 4x4 matrix representing the current head pose.

        """
        return self.client.get_current_head_pose()

    def _set_joint_positions(
        self,
        head_joint_positions: list[float] | None = None,
        antennas_joint_positions: list[float] | None = None,
    ) -> None:
        """Set the joint positions of the head and/or antennas.

        [Internal] Set the joint positions of the head and/or antennas.

        Args:
            head_joint_positions (Optional[List[float]]): List of head joint positions in radians (length 7).
            antennas_joint_positions (Optional[List[float]]): List of antennas joint positions in radians (length 2).
            record (Optional[Dict]): If provided, the command will be logged with the given record data.

        """
        cmd = {}

        if head_joint_positions is not None:
            assert len(head_joint_positions) == 7, (
                f"Head joint positions must have length 7, got {head_joint_positions}."
            )
            cmd["head_joint_positions"] = list(head_joint_positions)

        if antennas_joint_positions is not None:
            assert len(antennas_joint_positions) == 2, "Antennas must have length 2."
            cmd["antennas_joint_positions"] = list(antennas_joint_positions)

        if not cmd:
            raise ValueError(
                "At least one of head_joint_positions or antennas must be provided."
            )

        self.client.send_command(json.dumps(cmd))

    def set_target_head_pose(self, pose: npt.NDArray[np.float64]) -> None:
        """Set the head pose to a specific 4x4 matrix.

        Args:
            pose (np.ndarray): A 4x4 matrix representing the desired head pose.
            body_yaw (float): The yaw angle of the body, used to adjust the head pose.

        Raises:
            ValueError: If the shape of the pose is not (4, 4).

        """
        cmd = {}

        if pose is not None:
            assert pose.shape == (
                4,
                4,
            ), f"Head pose should be a 4x4 matrix, got {pose.shape}."
            cmd["head_pose"] = pose.tolist()
        else:
            raise ValueError("Pose must be provided as a 4x4 matrix.")

        self.client.send_command(json.dumps(cmd))

    def set_target_antenna_joint_positions(self, antennas: List[float]) -> None:
        """Set the target joint positions of the antennas."""
        cmd = {"antennas_joint_positions": antennas}
        self.client.send_command(json.dumps(cmd))

    def set_target_body_yaw(self, body_yaw: float) -> None:
        """Set the target body yaw.

        Args:
            body_yaw (float): The yaw angle of the body in radians.

        """
        cmd = {"body_yaw": body_yaw}
        self.client.send_command(json.dumps(cmd))

    def start_recording(self) -> None:
        """Start recording data."""
        self.client.send_command(json.dumps({"start_recording": True}))
        self.is_recording = True

    def stop_recording(
        self,
    ) -> Optional[List[Dict[str, float | List[float] | List[List[float]]]]]:
        """Stop recording data and return the recorded data."""
        self.client.send_command(json.dumps({"stop_recording": True}))
        self.is_recording = False
        if not self.client.wait_for_recorded_data(timeout=5):
            raise RuntimeError("Daemon did not provide recorded data in time!")
        recorded_data = self.client.get_recorded_data(wait=False)

        return recorded_data

    def _set_record_data(
        self, record: Dict[str, float | List[float] | List[List[float]]]
    ) -> None:
        """Set the record data to be logged by the backend.

        Args:
            record (Dict): The record data to be logged.

        """
        if not isinstance(record, dict):
            raise ValueError("Record must be a dictionary.")

        # Send the record data to the backend
        self.client.send_command(json.dumps({"set_target_record": record}))

    def enable_motors(self, ids: List[str] | None = None) -> None:
        """Enable the motors.

        Args:
            ids (List[str] | None): List of motor names to enable. If None, all motors will be enabled.
                Valid names match `src/reachy_mini/assets/config/hardware_config.yaml`:
                `body_rotation`, `stewart_1` … `stewart_6`, `right_antenna`, `left_antenna`.

        """
        self._set_torque(True, ids=ids)

    def disable_motors(self, ids: List[str] | None = None) -> None:
        """Disable the motors.

        Args:
            ids (List[str] | None): List of motor names to disable. If None, all motors will be disabled.
                Valid names match `src/reachy_mini/assets/config/hardware_config.yaml`:
                `body_rotation`, `stewart_1` … `stewart_6`, `right_antenna`, `left_antenna`.

        """
        self._set_torque(False, ids=ids)

    def _set_torque(self, on: bool, ids: List[str] | None = None) -> None:
        self.client.send_command(json.dumps({"torque": on, "ids": ids}))

    def enable_gravity_compensation(self) -> None:
        """Enable gravity compensation for the head motors."""
        self.client.send_command(json.dumps({"gravity_compensation": True}))

    def disable_gravity_compensation(self) -> None:
        """Disable gravity compensation for the head motors."""
        self.client.send_command(json.dumps({"gravity_compensation": False}))

    def set_automatic_body_yaw(self, body_yaw: float) -> None:
        """Set the automatic body yaw.

        Args:
            body_yaw (float): The yaw angle of the body in radians.

        """
        self.client.send_command(json.dumps({"automatic_body_yaw": body_yaw}))

    async def async_play_move(
        self,
        move: Move,
        play_frequency: float = 100.0,
        initial_goto_duration: float = 0.0,
        sound: bool = True,
    ) -> None:
        """Asynchronously play a Move.

        Args:
            move (Move): The Move object to be played.
            play_frequency (float): The frequency at which to evaluate the move (in Hz).
            initial_goto_duration (float): Duration for the initial goto to the starting position of the move (in seconds). If 0, no initial goto is performed.
            sound (bool): If True, play the sound associated with the move (if any).

        """
        if initial_goto_duration > 0.0:
            start_head_pose, start_antennas_positions, start_body_yaw = move.evaluate(
                0.0
            )
            self.goto_target(
                head=start_head_pose,
                antennas=start_antennas_positions,
                duration=initial_goto_duration,
                body_yaw=start_body_yaw,
            )

        sleep_period = 1.0 / play_frequency

        if move.sound_path is not None and sound:
            self.media_manager.play_sound(str(move.sound_path))

        t0 = time.time()
        while time.time() - t0 < move.duration:
            t = min(time.time() - t0, move.duration - 1e-2)

            head, antennas, body_yaw = move.evaluate(t)
            if head is not None:
                self.set_target_head_pose(head)
            if body_yaw is not None:
                self.set_target_body_yaw(body_yaw)
            if antennas is not None:
                self.set_target_antenna_joint_positions(list(antennas))

            elapsed = time.time() - t0 - t
            if elapsed < sleep_period:
                await asyncio.sleep(sleep_period - elapsed)
            else:
                await asyncio.sleep(0.001)

    play_move = async_to_sync(async_play_move)
