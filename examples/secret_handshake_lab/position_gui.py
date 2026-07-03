"""Position GUI with torque toggles: move the robot by hand, read the numbers.

Based on examples/mini_head_position_gui.py, extended for the antenna
collision investigation: buttons turn torque OFF/ON (all, or antennas only)
while the present position of everything keeps streaming, so you can move
the antennas by hand and read the exact angles at which they collide.

- "Head torque" button: toggles body_rotation + stewart_1..6.
- "Antennas torque" button: toggles left_antenna + right_antenna.
  With torque off, the sliders for that part stop driving the robot.
  Re-enabling is snap-free: enable_motors() pins targets to the present
  pose, and the sliders are synced to the present position first.
- The readout shows present antenna angles (rad and deg), diff and sum,
  head xyz / roll-pitch-yaw, and body yaw, refreshed at 50 Hz.
- "Snapshot" button (or Spacebar) prints one CSV line to the terminal and
  appends it to antenna_snapshots.csv next to this script: press it at
  interesting configurations (just touching, pressed, crossed, ...) to
  build the dataset for the geometric collision law.

Run (robot on):
    python examples/secret_handshake_lab/position_gui.py
"""

from __future__ import annotations

import os
import sys
import time

# uv-installed standalone Pythons bundle Tcl/Tk but do not find it on their
# own (TclError: "Can't find a usable init.tcl"). Point the env at the
# bundled copy before Tk starts.
if "TCL_LIBRARY" not in os.environ:
    _tcl = os.path.join(sys.base_prefix, "lib", "tcl8.6")
    _tk = os.path.join(sys.base_prefix, "lib", "tk8.6")
    if os.path.exists(os.path.join(_tcl, "init.tcl")):
        os.environ["TCL_LIBRARY"] = _tcl
        os.environ["TK_LIBRARY"] = _tk

import tkinter as tk

import numpy as np
from scipy.spatial.transform import Rotation as R

from collision import CollisionDetector

from reachy_mini import ReachyMini

HEAD_MOTORS = [
    "body_rotation",
    "stewart_1",
    "stewart_2",
    "stewart_3",
    "stewart_4",
    "stewart_5",
    "stewart_6",
]
ANTENNA_MOTORS = ["right_antenna", "left_antenna"]

SNAPSHOT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "antenna_snapshots.csv")
SNAPSHOT_HEADER = (
    "stamp,ant0_left_rad,ant1_right_rad,diff_rad,sum_rad,"
    "x_m,y_m,z_m,roll_deg,pitch_deg,yaw_deg,body_yaw_rad,note"
)


def main() -> None:
    with ReachyMini(media_backend="no_media") as mini:
        root = tk.Tk()
        root.title("Reachy Mini: positions + torque toggles")

        # ------------------------------------------------------------------
        # Target sliders (drive the robot only while that part has torque)
        # ------------------------------------------------------------------
        sliders_frame = tk.LabelFrame(root, text="Targets (active while torque ON)")
        sliders_frame.grid(row=0, column=0, padx=8, pady=4, sticky="nsew")

        def add_scale(frame, row, label, var, lo, hi, res=1.0):
            tk.Label(frame, text=label).grid(row=row, column=0, sticky="w")
            tk.Scale(
                frame, variable=var, from_=lo, to=hi, resolution=res,
                orient=tk.HORIZONTAL, length=220,
            ).grid(row=row, column=1)

        roll_var = tk.DoubleVar(value=0.0)
        pitch_var = tk.DoubleVar(value=0.0)
        yaw_var = tk.DoubleVar(value=0.0)
        x_var = tk.DoubleVar(value=0.0)
        y_var = tk.DoubleVar(value=0.0)
        z_var = tk.DoubleVar(value=0.0)
        body_yaw_var = tk.DoubleVar(value=0.0)
        ant_left_var = tk.DoubleVar(value=0.0)
        ant_right_var = tk.DoubleVar(value=0.0)

        add_scale(sliders_frame, 0, "Roll (deg):", roll_var, -45, 45)
        add_scale(sliders_frame, 1, "Pitch (deg):", pitch_var, -45, 45)
        add_scale(sliders_frame, 2, "Yaw (deg):", yaw_var, -175, 175)
        add_scale(sliders_frame, 3, "X (m):", x_var, -0.05, 0.05, 0.001)
        add_scale(sliders_frame, 4, "Y (m):", y_var, -0.05, 0.05, 0.001)
        add_scale(sliders_frame, 5, "Z (m):", z_var, -0.05, 0.03, 0.001)
        add_scale(sliders_frame, 6, "Body yaw (deg):", body_yaw_var, -180, 180)
        add_scale(sliders_frame, 7, "Antenna 0 / left (deg):", ant_left_var, -180, 180)
        add_scale(sliders_frame, 8, "Antenna 1 / right (deg):", ant_right_var, -180, 180)

        # ------------------------------------------------------------------
        # Torque toggles + snapshot
        # ------------------------------------------------------------------
        controls = tk.Frame(root)
        controls.grid(row=1, column=0, padx=8, pady=4, sticky="we")

        head_torque_on = tk.BooleanVar(value=True)
        antennas_torque_on = tk.BooleanVar(value=True)

        def sync_head_sliders() -> None:
            """Set head/body sliders to the present pose (no snap on enable)."""
            pose = mini.get_current_head_pose()
            rpy = R.from_matrix(np.array(pose)[:3, :3]).as_euler("xyz")
            roll_var.set(round(float(np.rad2deg(rpy[0])), 1))
            pitch_var.set(round(float(np.rad2deg(rpy[1])), 1))
            yaw_var.set(round(float(np.rad2deg(rpy[2])), 1))
            x_var.set(round(float(pose[0][3]), 3))
            y_var.set(round(float(pose[1][3]), 3))
            z_var.set(round(float(pose[2][3]), 3))
            head_joints, _ = mini.get_current_joint_positions()
            body_yaw_var.set(round(float(np.rad2deg(head_joints[0])), 1))

        def sync_antenna_sliders() -> None:
            ant0, ant1 = mini.get_present_antenna_joint_positions()
            ant_left_var.set(round(float(np.rad2deg(ant0)), 1))
            ant_right_var.set(round(float(np.rad2deg(ant1)), 1))

        def toggle(part: str) -> None:
            var = head_torque_on if part == "head" else antennas_torque_on
            motors = HEAD_MOTORS if part == "head" else ANTENNA_MOTORS
            if var.get():
                mini.disable_motors(ids=motors)
                var.set(False)
            else:
                if part == "head":
                    sync_head_sliders()
                else:
                    sync_antenna_sliders()
                mini.enable_motors(ids=motors)  # pins to present pose first
                var.set(True)
            refresh_buttons()

        def refresh_buttons() -> None:
            head_btn.config(
                text=f"Head torque: {'ON' if head_torque_on.get() else 'OFF'}",
                fg="white",
                bg="#2e7d32" if head_torque_on.get() else "#c62828",
                activebackground="#2e7d32" if head_torque_on.get() else "#c62828",
            )
            ant_btn.config(
                text=f"Antennas torque: {'ON' if antennas_torque_on.get() else 'OFF'}",
                fg="white",
                bg="#2e7d32" if antennas_torque_on.get() else "#c62828",
                activebackground="#2e7d32" if antennas_torque_on.get() else "#c62828",
            )

        head_btn = tk.Button(controls, command=lambda: toggle("head"))
        head_btn.grid(row=0, column=0, padx=4)
        ant_btn = tk.Button(controls, command=lambda: toggle("antennas"))
        ant_btn.grid(row=0, column=1, padx=4)
        snap_btn = tk.Button(controls, text="Snapshot (Space)")
        snap_btn.grid(row=0, column=2, padx=4)
        refresh_buttons()

        # ------------------------------------------------------------------
        # Live present-position readout
        # ------------------------------------------------------------------
        readout = tk.Label(
            root, font=("Courier", 13), justify=tk.LEFT, anchor="w"
        )
        readout.grid(row=2, column=0, padx=8, pady=6, sticky="we")

        latest = {"line": ""}  # last snapshot-formatted values

        def take_snapshot(_event=None) -> None:
            if not latest["line"]:
                return
            line = latest["line"]
            new_file = not os.path.exists(SNAPSHOT_CSV)
            with open(SNAPSHOT_CSV, "a") as f:
                if new_file:
                    f.write(SNAPSHOT_HEADER + "\n")
                f.write(line + "\n")
            print(line)

        snap_btn.config(command=take_snapshot)
        root.bind("<space>", take_snapshot)

        collision_det = CollisionDetector()

        def tick() -> None:
            ant0, ant1 = mini.get_present_antenna_joint_positions()
            collision_det.update(time.monotonic(), ant0, ant1)
            pose = np.array(mini.get_current_head_pose())
            head_joints, _ = mini.get_current_joint_positions()
            body_yaw = float(head_joints[0])
            rpy = np.rad2deg(R.from_matrix(pose[:3, :3]).as_euler("xyz"))
            x, y, z = pose[0, 3], pose[1, 3], pose[2, 3]

            readout.config(
                text=(
                    f"ANTENNAS (present)\n"
                    f"  left  (ant0): {ant0:+8.4f} rad  {np.rad2deg(ant0):+8.2f} deg\n"
                    f"  right (ant1): {ant1:+8.4f} rad  {np.rad2deg(ant1):+8.2f} deg\n"
                    f"  diff (a0-a1): {np.rad2deg(ant0 - ant1):+8.2f} deg  "
                    f"sum (a0+a1): {np.rad2deg(ant0 + ant1):+8.2f} deg\n"
                    f"  collision band [{collision_det.sum_lo_deg:.0f},"
                    f"{collision_det.sum_hi_deg:.0f}]: "
                    f"{'>>> IN COLLISION <<<' if collision_det.in_collision else '(not colliding)'}\n"
                    f"HEAD (present)\n"
                    f"  xyz: {x * 1000:+7.1f} {y * 1000:+7.1f} {z * 1000:+7.1f} mm\n"
                    f"  rpy: {rpy[0]:+7.1f} {rpy[1]:+7.1f} {rpy[2]:+7.1f} deg\n"
                    f"  body yaw: {np.rad2deg(body_yaw):+7.1f} deg"
                )
            )
            latest["line"] = (
                f"{time.time():.3f},{ant0:.5f},{ant1:.5f},"
                f"{ant0 - ant1:.5f},{ant0 + ant1:.5f},"
                f"{x:.4f},{y:.4f},{z:.4f},"
                f"{rpy[0]:.2f},{rpy[1]:.2f},{rpy[2]:.2f},{body_yaw:.4f},"
            )

            # Drive only the parts that have torque.
            head = None
            body = None
            antennas = None
            if head_torque_on.get():
                head = np.eye(4)
                head[:3, :3] = R.from_euler(
                    "xyz",
                    np.deg2rad([roll_var.get(), pitch_var.get(), yaw_var.get()]),
                ).as_matrix()
                head[:3, 3] = [x_var.get(), y_var.get(), z_var.get()]
                body = float(np.deg2rad(body_yaw_var.get()))
            if antennas_torque_on.get():
                antennas = np.deg2rad([ant_left_var.get(), ant_right_var.get()])
            if head is not None or antennas is not None:
                mini.set_target(head=head, antennas=antennas, body_yaw=body)

            root.after(30, tick)  # ~33 Hz, plenty for a readout + sliders

        # Start from the present pose so launching the GUI never moves the
        # robot: sliders are synced before the first set_target is sent.
        sync_head_sliders()
        sync_antenna_sliders()
        print("snapshot columns:", SNAPSHOT_HEADER)
        root.after(30, tick)

        try:
            root.mainloop()
        except KeyboardInterrupt:
            pass
        finally:
            try:
                root.destroy()
            except tk.TclError:
                pass
        print(f"\nsnapshots (if any) appended to {SNAPSHOT_CSV}")


if __name__ == "__main__":
    main()
