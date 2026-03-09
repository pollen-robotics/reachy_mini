"""MuJoCo Backend for Reachy Mini Daemon."""

from reachy_mini.io.protocol import MujocoBackendStatus

try:
    import mujoco  # noqa: F401

    from reachy_mini.daemon.backend.mujoco.backend import MujocoBackend

except ImportError:

    class MujocoMockupBackend:
        """Mockup class to avoid import errors when MuJoCo is not installed."""

        def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
            """Raise ImportError when trying to instantiate the class."""
            raise ImportError(
                "MuJoCo is not installed. MuJoCo backend is not available."
                " To use MuJoCo backend, please install the 'mujoco' extra dependencies"
                " with 'pip install reachy_mini[mujoco]'."
            )

    MujocoBackend = MujocoMockupBackend  # type: ignore[assignment, misc]

__all__ = ["MujocoBackend", "MujocoBackendStatus"]
