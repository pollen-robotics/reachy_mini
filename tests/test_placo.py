def test_load_kinematics():  # noqa: D100, D103
    from reachy_mini.reachy_mini import PlacoKinematics, ReachyMini

    # Test loading the kinematics
    kinematics = PlacoKinematics(ReachyMini.urdf_root_path)
    assert kinematics is not None, "Failed to load PlacoKinematics."
