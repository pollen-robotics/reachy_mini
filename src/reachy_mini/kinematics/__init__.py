try:  # noqa: D104
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

try:
    from reachy_mini.kinematics.placo_kinematics import PlacoKinematics  # noqa: F401
except ImportError:

    class MockupPlacoKinematics:
        """Mockup class for PlacoKinematics."""

        def __init__(self, *args, **kwargs):
            """Raise ImportError when trying to instantiate the class."""
            raise ImportError(
                "PlacoKinematics could not be imported. Make sure you run pip install reachy_mini[placo_kinematics]."
            )

    PlacoKinematics = MockupPlacoKinematics


from reachy_mini.kinematics.rust_kinematics import (  # noqa: F401
    RustKinematics,
)
