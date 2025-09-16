from reachy_mini.kinematics.analytic_kinematics import (  # noqa: D104, F401, I001
    ReachyMiniAnalyticKinematics,
)

try:
    from reachy_mini.kinematics.nn_kinematics import NNKinematics  # noqa: F401
except ImportError:

    class MockupNNKinematics:
        """Mockup class for NNKinematics."""

        def __init__(self, *args, **kwargs):
            """Raise ImportError when trying to instantiate the class."""
            raise ImportError(
                "NNKinematics could not be imported. Make sure you run pip install reachy_mini[nn_kinematics]."
            )

    NNKinematics = MockupNNKinematics


from reachy_mini.kinematics.cpp_analytic_kinematics import (
    CPPAnalyticKinematics,  # noqa: F401
)
from reachy_mini.kinematics.placo_kinematics import PlacoKinematics  # noqa: F401
from reachy_mini.kinematics.rust_kinematics import (  # noqa: F401
    RustKinematics,
)
