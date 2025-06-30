import time

import numpy as np

from reachy_mini import ReachyMini
import tkinter as tk


def main():
    with ReachyMini() as mini:
        t0 = time.time()
        while True:
            t = time.time() - t0

            target = np.deg2rad(30) * np.sin(2 * np.pi * 0.5 * t)

            # Create a simple Tkinter GUI to set Euler angles
            if not hasattr(main, "gui_initialized"):
                root = tk.Tk()
                root.title("Set Head Euler Angles")

                all_yaw_var = tk.DoubleVar(value=0.0)
                var_1 = tk.DoubleVar(value=0.0)
                var_2 = tk.DoubleVar(value=0.0)
                var_3 = tk.DoubleVar(value=0.0)
                var_4 = tk.DoubleVar(value=0.0)
                var_5 = tk.DoubleVar(value=0.0)
                var_6 = tk.DoubleVar(value=0.0)

                tk.Label(root, text="all_yaw (deg):").grid(row=0, column=0)
                tk.Scale(
                    root, variable=all_yaw_var, from_=-90, to=90, orient=tk.HORIZONTAL
                ).grid(row=0, column=1)
                tk.Label(root, text="1 (deg):").grid(row=1, column=0)
                tk.Scale(
                    root, variable=var_1, from_=-90, to=90, orient=tk.HORIZONTAL
                ).grid(row=1, column=1)
                tk.Label(root, text="2 (deg):").grid(row=2, column=0)
                tk.Scale(
                    root, variable=var_2, from_=-90, to=90, orient=tk.HORIZONTAL
                ).grid(row=2, column=1)
                tk.Label(root, text="3 (deg):").grid(row=3, column=0)
                tk.Scale(
                    root, variable=var_3, from_=-90, to=90, orient=tk.HORIZONTAL
                ).grid(row=3, column=1)
                tk.Label(root, text="4 (deg):").grid(row=4, column=0)
                tk.Scale(
                    root, variable=var_4, from_=-90, to=90, orient=tk.HORIZONTAL
                ).grid(row=4, column=1)
                tk.Label(root, text="5 (deg):").grid(row=5, column=0)
                tk.Scale(
                    root, variable=var_5, from_=-90, to=90, orient=tk.HORIZONTAL
                ).grid(row=5, column=1)
                tk.Label(root, text="6 (deg):").grid(row=6, column=0)
                tk.Scale(
                    root, variable=var_6, from_=-90, to=90, orient=tk.HORIZONTAL
                ).grid(row=6, column=1)

                # add a checkbox to enable/disable collision checking
                collision_check_var = tk.BooleanVar(value=False)
                tk.Checkbutton(
                    root, text="Check Collision", variable=collision_check_var
                ).grid(row=7, column=1)

                # Run the GUI in a non-blocking way
                root.update()
                main.gui_initialized = True

            root.update()

            try:
                mini._send_joint_command(
                    head_joint_positions=[
                        np.deg2rad(all_yaw_var.get()),
                        np.deg2rad(var_1.get()),
                        np.deg2rad(var_2.get()),
                        np.deg2rad(var_3.get()),
                        np.deg2rad(var_4.get()),
                        np.deg2rad(var_5.get()),
                        np.deg2rad(var_6.get()),
                    ],
                    antennas_joint_positions=[target, target],
                    check_collision=collision_check_var.get(),
                )
            except ValueError as e:
                print(f"Error: {e}")
                continue

            # print(mini.get_manip())
            # print(mini.get_collision())
            time.sleep(0.01)


if __name__ == "__main__":
    main()
