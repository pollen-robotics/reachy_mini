# Head Mass Torque Test

Test motor torques at various head poses with incrementally increasing head mass.

## Goal

Find the maximum safe head mass before reaching 75% of motor stall torque (0.45 N·m limit) by modifying the head mass in the MJCF XML file.

## Features

- Tests head mass in 50g increments starting from original mass
- Tests pitch, roll, and translation poses
- Modifies XML file to change head mass (not center of mass)
- Stops when any motor exceeds safe torque limit
- Generates robot renders and torque graphs for each test
- Creates summary visualization showing torque vs mass curves

## Usage

```bash
cd head_mass_torque_test
python check_head_torques.py
```

This will:
1. Read original head mass from `reachy_mini.xml`
2. Test all poses at each mass increment (50g steps)
3. Create temporary modified XML files for each mass
4. Stop when torque limit is reached
5. Save renders and graphs to `head_torque_check_results/`
6. Generate summary figure with torque vs mass curves

## Test Poses

13 poses tested for each mass:

### Rotation Poses
- Neutral (0°, 0°)
- Pitch up 15° and 30° (head looking up)
- Pitch down 15° and 30° (head looking down)
- Roll left 15° and 30°
- Roll right 15° and 30°

### Translation Poses
- Translate up 2cm
- Translate down 1cm
- Translate forward 1cm
- Translate side 1cm

## Configuration

### Torque Limit
```python
SAFE_LIMIT_NM = 0.45  # 75% of 0.6 N·m stall torque
STALL_TORQUE_NM = 0.6
```

### Mass Testing
```python
MASS_INCREMENT_KG = 0.05  # 50g steps
MAX_MASS_KG = 2.0  # Safety upper limit
```

## Output

Results saved to `head_torque_check_results/`:
- `{pose}_{mass}g_render.png` - MuJoCo render for each pose/mass
- `{pose}_{mass}g_torques.png` - Motor torque bar chart for each pose/mass
- `mass_test_summary.png` - Summary showing torque vs mass curves and max safe mass per pose

## Notes

- Roll direction fixed: positive roll = right, negative roll = left (standard convention)
- Head mass is modified in the XML inertial element (not COM position)
- Original XML file is never modified (uses temporary copies)
- Tests stop immediately when any motor exceeds 0.45 N·m
