# Maximum Payload Per Pose

This experiment finds the maximum payload mass for **each individual pose** before reaching the safe torque limit (0.45 N·m).

## Two Configurations

### 1. Inside Head
- Modifies `xl_330` body mass (the head itself)
- Scene: `empty`
- Simulates payload integrated within the head structure

### 2. On Top of Head
- Modifies `fake_payload_head` body mass (red box on top)
- Scene: `empty_payload`
- Simulates payload mounted externally on top of head

## Method

Uses **binary search** to efficiently find maximum mass for each pose:
1. Start with range [0g, 2000g]
2. Test middle mass
3. If torque < safe limit: try higher mass
4. If torque ≥ safe limit: try lower mass
5. Repeat until range is within 5g tolerance

## Usage

```bash
cd max_payload_per_pose

# Test both configurations
python3 find_max_payload_per_pose.py --mode both

# Test only inside head
python3 find_max_payload_per_pose.py --mode inside

# Test only on top of head
python3 find_max_payload_per_pose.py --mode ontop
```

## Output

- `max_payload_inside.npy` - Results for inside head configuration
- `max_payload_ontop.npy` - Results for on top configuration
- `max_payload_comparison.png` - Side-by-side comparison chart

## Results

The script will show:
- Maximum safe payload for each pose
- Most/least restrictive poses
- Average maximum payload across all poses
- Comparison between inside and on-top configurations

## Notes

- Safe torque limit: 0.45 N·m (75% of 0.6 N·m stall torque)
- Tests 9 poses (neutral, pitch ±15°/±30°, roll ±15°/±30°)
- Binary search converges within ~10-15 iterations per pose
