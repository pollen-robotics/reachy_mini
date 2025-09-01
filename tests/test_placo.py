def test_load_kinematics():  # noqa: D100, D103
    from reachy_mini.daemon.backend.abstract import Backend
    from reachy_mini.kinematics import PlacoKinematics

    # Test loading the kinematics
    kinematics = PlacoKinematics(Backend.urdf_root_path)
    assert kinematics is not None, "Failed to load PlacoKinematics."
