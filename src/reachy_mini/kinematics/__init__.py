"""Try to import kinematics engines, and provide mockup classes if they are not available.

``NNKinematics`` is exposed lazily (PEP 562): importing it eagerly pulls
onnxruntime (~1-1.5s of CPU on the wireless robot) into every process that
touches this package — including the daemon, which defaults to
``AnalyticalKinematics`` and would pay for onnxruntime at boot for nothing.
"""

from typing import TYPE_CHECKING, Annotated, Any

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from reachy_mini.kinematics.nn_kinematics import NNKinematics

try:
    from reachy_mini.kinematics.placo_kinematics import PlacoKinematics  # noqa: F401
except ImportError:

    class MockupPlacoKinematics:
        """Mockup class for PlacoKinematics."""

        def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
            """Raise ImportError when trying to instantiate the class."""
            raise ImportError(
                "PlacoKinematics could not be imported. Make sure you run pip install reachy_mini[placo_kinematics]."
            )

        def ik(self, *args, **kwargs) -> Annotated[npt.NDArray[np.float64], (7,)]:  # type: ignore[no-untyped-def]
            """Mockup method for ik."""
            raise ImportError(
                "PlacoKinematics could not be imported. Make sure you run pip install reachy_mini[placo_kinematics]."
            )

        def fk(self, *args, **kwargs) -> Annotated[npt.NDArray[np.float64], (4, 4)]:  # type: ignore[no-untyped-def]
            """Mockup method for fk."""
            raise ImportError(
                "PlacoKinematics could not be imported. Make sure you run pip install reachy_mini[placo_kinematics]."
            )

    PlacoKinematics = MockupPlacoKinematics  # type: ignore[assignment, misc]


from reachy_mini.kinematics.analytical_kinematics import (  # noqa: F401
    AnalyticalKinematics,
)

if TYPE_CHECKING:
    AnyKinematics = NNKinematics | PlacoKinematics | AnalyticalKinematics

__all__ = ["NNKinematics", "PlacoKinematics", "AnalyticalKinematics"]


def __getattr__(name: str) -> Any:
    """Lazily resolve ``NNKinematics`` so the package import stays onnxruntime-free."""
    if name == "NNKinematics":
        from reachy_mini.kinematics.nn_kinematics import NNKinematics

        return NNKinematics
    if name == "AnyKinematics":
        from reachy_mini.kinematics.nn_kinematics import NNKinematics

        return NNKinematics | PlacoKinematics | AnalyticalKinematics
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
