"""Connection manager for WebRTC peers."""

import asyncio
import json
import logging
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R

from reachy_mini.daemon.backend.abstract import MotorControlMode
from reachy_mini.daemon.webrtc.peer import WebRTCPeer

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages multiple WebRTC peer connections."""

    def __init__(self) -> None:
        """Initialize the connection manager."""
        self._peers: dict[str, WebRTCPeer] = {}
        self._message_handler: MessageHandler | None = None

    def set_message_handler(self, handler: "MessageHandler") -> None:
        """Set the message handler for processing incoming commands.

        Args:
            handler: The message handler instance
        """
        self._message_handler = handler

    async def create_peer(self, peer_id: str) -> WebRTCPeer:
        """Create a new peer connection.

        Args:
            peer_id: Unique identifier for the peer

        Returns:
            The created WebRTCPeer instance
        """
        if peer_id in self._peers:
            await self._peers[peer_id].close()

        peer = WebRTCPeer(
            peer_id=peer_id,
            on_message=self._handle_message,
            on_close=self._on_peer_close
        )
        self._peers[peer_id] = peer
        logger.info(f"Created peer {peer_id}, total peers: {len(self._peers)}")
        return peer

    def get_peer(self, peer_id: str) -> WebRTCPeer | None:
        """Get an existing peer by ID.

        Args:
            peer_id: The peer ID to look up

        Returns:
            The peer if found, None otherwise
        """
        return self._peers.get(peer_id)

    async def remove_peer(self, peer_id: str) -> None:
        """Remove and close a peer connection.

        Args:
            peer_id: The peer ID to remove
        """
        if peer_id in self._peers:
            await self._peers[peer_id].close()
            del self._peers[peer_id]
            logger.info(f"Removed peer {peer_id}, remaining: {len(self._peers)}")

    def _on_peer_close(self, peer_id: str) -> None:
        """Handle peer close event.

        Args:
            peer_id: The ID of the closed peer
        """
        if peer_id in self._peers:
            del self._peers[peer_id]
            logger.info(f"Peer {peer_id} closed, remaining: {len(self._peers)}")

    async def _handle_message(self, peer_id: str, message: str) -> str:
        """Handle incoming message from a peer.

        Args:
            peer_id: The peer that sent the message
            message: The message content

        Returns:
            Response string to send back
        """
        if self._message_handler:
            return await self._message_handler.handle(peer_id, message)
        return json.dumps({"error": "No message handler configured", "type": "error"})

    def broadcast(self, message: str) -> int:
        """Broadcast a message to all connected peers.

        Args:
            message: The message to broadcast

        Returns:
            Number of peers the message was sent to
        """
        sent = 0
        for peer in self._peers.values():
            if peer.send(message):
                sent += 1
        return sent

    @property
    def peer_count(self) -> int:
        """Get the number of connected peers."""
        return len(self._peers)

    @property
    def peer_ids(self) -> list[str]:
        """Get list of connected peer IDs."""
        return list(self._peers.keys())

    async def close_all(self) -> None:
        """Close all peer connections."""
        for peer_id in list(self._peers.keys()):
            await self.remove_peer(peer_id)


class MessageHandler:
    """Handles incoming WebRTC messages and routes them to API handlers."""

    def __init__(self, backend_getter: Any) -> None:
        """Initialize message handler.

        Args:
            backend_getter: Callable or object to get the backend instance
        """
        self._backend_getter = backend_getter
        self._handlers: dict[str, Any] = {}

    async def handle(self, peer_id: str, message: str) -> str:
        """Handle an incoming message and return response.

        Args:
            peer_id: The peer that sent the message
            message: JSON-encoded message

        Returns:
            JSON-encoded response
        """
        try:
            data = json.loads(message)
        except json.JSONDecodeError as e:
            return json.dumps({
                "error": f"Invalid JSON: {e}",
                "type": "error"
            })

        action = data.get("action", "")
        params = data.get("params", {})
        request_id = data.get("id")

        logger.debug(f"Handling action '{action}' from peer {peer_id}")

        try:
            result = await self._route_action(action, params)
            response = {
                "type": "response",
                "action": action,
                "data": result
            }
            if request_id:
                response["id"] = request_id
            return json.dumps(response)

        except Exception as e:
            logger.error(f"Error handling action '{action}': {e}")
            error_response = {
                "type": "error",
                "action": action,
                "error": str(e)
            }
            if request_id:
                error_response["id"] = request_id
            return json.dumps(error_response)

    async def _route_action(self, action: str, params: dict[str, Any]) -> Any:
        """Route an action to the appropriate handler.

        Args:
            action: The action name (e.g., "state/full", "move/goto")
            params: Action parameters

        Returns:
            The result from the handler
        """
        backend = self._get_backend()
        if backend is None:
            raise RuntimeError("Backend not available")

        # Route based on action prefix
        if action.startswith("state/"):
            return await self._handle_state(action, params, backend)
        elif action.startswith("move/"):
            return await self._handle_move(action, params, backend)
        elif action.startswith("motors/"):
            return await self._handle_motors(action, params, backend)
        elif action.startswith("joints/"):
            return await self._handle_joints(action, params, backend)
        elif action.startswith("pose/"):
            return await self._handle_pose(action, params, backend)
        elif action == "ping":
            return {"pong": True}
        else:
            raise ValueError(f"Unknown action: {action}")

    def _get_backend(self) -> Any:
        """Get the backend instance."""
        if callable(self._backend_getter):
            return self._backend_getter()
        return self._backend_getter

    async def _handle_state(self, action: str, params: dict[str, Any], backend: Any) -> Any:
        """Handle state-related actions."""
        if action == "state/full":
            head_pose = backend.get_present_head_pose()
            return {
                "head_pose": head_pose.tolist() if hasattr(head_pose, 'tolist') else head_pose,
                "body_yaw": backend.get_present_body_yaw(),
                "antenna_positions": list(backend.get_present_antenna_joint_positions()),
                "control_mode": backend.get_motor_control_mode().value,
            }
        elif action == "state/head_pose":
            pose = backend.get_present_head_pose()
            return pose.tolist() if hasattr(pose, 'tolist') else pose
        elif action == "state/body_yaw":
            return backend.get_present_body_yaw()
        elif action == "state/antenna_positions":
            return list(backend.get_present_antenna_joint_positions())
        elif action == "state/joints":
            head_joints = backend.get_present_head_joint_positions()
            antenna_joints = backend.get_present_antenna_joint_positions()
            return {
                "head_joints": list(head_joints),
                "antenna_joints": list(antenna_joints),
            }
        else:
            raise ValueError(f"Unknown state action: {action}")

    async def _handle_move(self, action: str, params: dict[str, Any], backend: Any) -> Any:
        """Handle movement-related actions."""
        if action == "move/goto":
            duration = params.get("duration", 1.0)

            # Parse target parameters
            head_pose = params.get("head_pose")
            antennas = params.get("antennas")
            body_yaw = params.get("body_yaw")

            # Convert head_pose to numpy array if provided
            head_array = None
            if head_pose is not None:
                if isinstance(head_pose, list):
                    head_array = np.array(head_pose)
                elif isinstance(head_pose, dict):
                    # Convert x,y,z,roll,pitch,yaw to pose matrix
                    from reachy_mini.daemon.app.models import AnyPose
                    pose = AnyPose.model_validate(head_pose)
                    head_array = pose.to_pose_array()

            # Convert antennas to numpy array if provided
            antennas_array = np.array(antennas) if antennas is not None else None

            # Execute the goto
            await backend.goto_target(
                head=head_array,
                antennas=antennas_array,
                body_yaw=body_yaw,
                duration=duration,
            )
            return {"status": "ok"}

        elif action == "move/set_target":
            target = params.get("target")
            if target is None:
                raise ValueError("Missing 'target' parameter")

            backend.set_target(target)
            return {"status": "ok"}

        elif action == "move/wake_up":
            await backend.wake_up()
            return {"status": "ok"}

        elif action == "move/goto_sleep":
            await backend.goto_sleep()
            return {"status": "ok"}

        elif action == "move/stop":
            # Note: stop functionality would need task tracking
            return {"status": "ok"}

        else:
            raise ValueError(f"Unknown move action: {action}")

    async def _handle_motors(self, action: str, params: dict[str, Any], backend: Any) -> Any:
        """Handle motor-related actions."""
        if action == "motors/status":
            return {"mode": backend.get_motor_control_mode().value}

        elif action == "motors/set_mode":
            mode_str = params.get("mode")
            if mode_str is None:
                raise ValueError("Missing 'mode' parameter")

            # Convert string to MotorControlMode enum
            mode_map = {
                "enabled": MotorControlMode.Enabled,
                "disabled": MotorControlMode.Disabled,
                "gravity_compensation": MotorControlMode.GravityCompensation,
            }

            if mode_str not in mode_map:
                raise ValueError(f"Invalid mode: {mode_str}. Must be one of: {list(mode_map.keys())}")

            backend.set_motor_control_mode(mode_map[mode_str])
            return {"status": "ok", "mode": mode_str}

        else:
            raise ValueError(f"Unknown motors action: {action}")

    async def _handle_joints(self, action: str, params: dict[str, Any], backend: Any) -> Any:
        """Handle joint-related actions for direct control."""
        if action == "joints/set_target":
            head_joints = params.get("head_joints")
            antenna_joints = params.get("antenna_joints")

            if head_joints is not None:
                backend.set_target_head_joint_positions(np.array(head_joints))

            if antenna_joints is not None:
                backend.set_target_antenna_joint_positions(np.array(antenna_joints))

            return {"status": "ok"}

        elif action == "joints/get":
            head_joints = backend.get_present_head_joint_positions()
            antenna_joints = backend.get_present_antenna_joint_positions()
            return {
                "head_joints": list(head_joints),
                "antenna_joints": list(antenna_joints),
            }

        else:
            raise ValueError(f"Unknown joints action: {action}")

    async def _handle_pose(self, action: str, params: dict[str, Any], backend: Any) -> Any:
        """Handle Cartesian pose control for intuitive head control."""
        if action == "pose/set_target":
            # Get head pose parameters (x, y, z, roll, pitch, yaw)
            head_pose = params.get("head_pose")
            body_yaw = params.get("body_yaw")
            antennas = params.get("antennas")

            # Convert x,y,z,roll,pitch,yaw to 4x4 matrix
            head_matrix = None
            if head_pose is not None:
                x = head_pose.get("x", 0.0)
                y = head_pose.get("y", 0.0)
                z = head_pose.get("z", 0.0)
                roll = head_pose.get("roll", 0.0)
                pitch = head_pose.get("pitch", 0.0)
                yaw = head_pose.get("yaw", 0.0)

                rotation = R.from_euler("xyz", [roll, pitch, yaw])
                head_matrix = np.eye(4)
                head_matrix[:3, 3] = [x, y, z]
                head_matrix[:3, :3] = rotation.as_matrix()

            # Set targets
            backend.set_target(
                head=head_matrix,
                body_yaw=body_yaw,
                antennas=np.array(antennas) if antennas is not None else None
            )

            return {"status": "ok"}

        elif action == "pose/get":
            # Get current pose in x,y,z,roll,pitch,yaw format
            head_pose = backend.get_present_head_pose()
            body_yaw = backend.get_present_body_yaw()
            antennas = backend.get_present_antenna_joint_positions()

            # Convert 4x4 matrix to x,y,z,roll,pitch,yaw
            x, y, z = head_pose[0, 3], head_pose[1, 3], head_pose[2, 3]
            rotation = R.from_matrix(head_pose[:3, :3])
            roll, pitch, yaw = rotation.as_euler("xyz")

            return {
                "head_pose": {
                    "x": float(x),
                    "y": float(y),
                    "z": float(z),
                    "roll": float(roll),
                    "pitch": float(pitch),
                    "yaw": float(yaw)
                },
                "body_yaw": float(body_yaw),
                "antennas": list(antennas)
            }

        else:
            raise ValueError(f"Unknown pose action: {action}")
