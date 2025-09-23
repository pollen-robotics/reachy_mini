def test_load_kinematics():  # noqa: D100, D103
    from reachy_mini.utils.constants import URDF_ROOT_PATH
    from reachy_mini.kinematics import PlacoKinematics

    # Test loading the kinematics
    kinematics = PlacoKinematics(URDF_ROOT_PATH)
    assert kinematics is not None, "Failed to load PlacoKinematics."
