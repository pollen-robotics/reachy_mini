"""Kinematics engines for Reachy Mini.

PlacoKinematics is the default and required engine for proper passive joint calculation.
AnalyticalKinematics is available as a fallback.
NNKinematics is optional and requires pip install reachy_mini[nn_kinematics].
"""

from typing import Annotated

import numpy as np
import numpy.typing as npt

# PlacoKinematics is now a core dependency - import directly
from reachy_mini.kinematics.placo_kinematics import PlacoKinematics  # noqa: F401

# AnalyticalKinematics is always available
from reachy_mini.kinematics.analytical_kinematics import (  # noqa: F401
    AnalyticalKinematics,
)

# NNKinematics is optional
try:
    from reachy_mini.kinematics.nn_kinematics import NNKinematics  # noqa: F401
except ImportError:

    class MockupNNKinematics:
        """Mockup class for NNKinematics."""

        def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
            """Raise ImportError when trying to instantiate the class."""
            raise ImportError(
                "NNKinematics could not be imported. Make sure you run pip install reachy_mini[nn_kinematics]."
            )

        def ik(self, *args, **kwargs) -> Annotated[npt.NDArray[np.float64], (7,)]:  # type: ignore[no-untyped-def]
            """Mockup method for ik."""
            raise ImportError(
                "NNKinematics could not be imported. Make sure you run pip install reachy_mini[nn_kinematics]."
            )

        def fk(self, *args, **kwargs) -> Annotated[npt.NDArray[np.float64], (4, 4)]:  # type: ignore[no-untyped-def]
            """Mockup method for fk."""
            raise ImportError(
                "NNKinematics could not be imported. Make sure you run pip install reachy_mini[nn_kinematics]."
            )

    NNKinematics = MockupNNKinematics  # type: ignore[assignment, misc]


AnyKinematics = NNKinematics | PlacoKinematics | AnalyticalKinematics
__all__ = ["NNKinematics", "PlacoKinematics", "AnalyticalKinematics"]
