def test_load_kinematics():
    from reachy_mini.reachy_mini import PlacoKinematics, ReachyMini

    # Test loading the kinematics
    kinematics = PlacoKinematics(ReachyMini.urdf_root_path)
    assert kinematics is not None, "Failed to load PlacoKinematics."
