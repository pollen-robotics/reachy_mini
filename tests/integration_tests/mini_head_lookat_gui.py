"""Reachy Mini Head Position GUI Example."""

import time
import tkinter as tk

import numpy as np

from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose


def main():
    """Run a GUI to set the head position and orientation of Reachy Mini."""
    with ReachyMini(media_backend="no_media") as mini:
        t0 = time.time()

        root = tk.Tk()
        root.title("Set Look At XYZ Position")

        # Add sliders for X, Y, Z position
        x_var = tk.DoubleVar(value=0.0)
        y_var = tk.DoubleVar(value=0.0)
        z_var = tk.DoubleVar(value=0.0)

        tk.Label(root, text="X (m):").grid(row=0, column=0)
        tk.Scale(
            root,
            variable=x_var,
            from_=-0.2,
            to=0.2,
            resolution=0.001,
            orient=tk.HORIZONTAL,
            length=200,
        ).grid(row=0, column=1)
        tk.Label(root, text="Y (m):").grid(row=1, column=0)
        tk.Scale(
            root,
            variable=y_var,
            from_=-0.2,
            to=0.2,
            resolution=0.001,
            orient=tk.HORIZONTAL,
            length=200,
        ).grid(row=1, column=1)
        tk.Label(root, text="Z (m):").grid(row=2, column=0)
        tk.Scale(
            root,
            variable=z_var,
            from_=-0.2,
            to=0.2,
            resolution=0.001,
            orient=tk.HORIZONTAL,
            length=200,
        ).grid(row=2, column=1)

        tk.Label(root, text="Body Yaw (deg):").grid(row=3, column=0)
        body_yaw_var = tk.DoubleVar(value=0.0)
        tk.Scale(
            root,
            variable=body_yaw_var,
            from_=-180,
            to=180,
            resolution=1.0,
            orient=tk.HORIZONTAL,
            length=200,
        ).grid(row=3, column=1)

        mini.goto_target(create_head_pose(), antennas=[0.0, 0.0], duration=1.0)

        # Run the GUI in a non-blocking way
        root.update()

        try:
            while True:
                t = time.time() - t0
                target = np.deg2rad(30) * np.sin(2 * np.pi * 0.5 * t)

                head = np.eye(4)
                head[:3, 3] = [0, 0, 0.0]

                head = mini.look_at_world(
                    x_var.get(),
                    y_var.get(),
                    z_var.get(),
                    duration=0.3,
                    perform_movement=False,
                )

                root.update()

                mini.set_target(
                    head=head,
                    body_yaw=np.deg2rad(body_yaw_var.get()),
                    antennas=np.array([target, -target]),
                )

        except KeyboardInterrupt:
            pass
        finally:
            root.destroy()


if __name__ == "__main__":
    main()
