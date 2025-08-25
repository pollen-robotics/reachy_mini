import numpy as np


current_position = 0
target_position = 0
dt = 0.01


def s(t):
    return 1 if t < 3 else max(0, (5 - t) / (5 - 3))


def d_s(t):
    return s(t + dt) - s(t)


def e(t):
    # return 3 * np.sin(t * 5)
    return 3 * np.sin(t * 5 + 1)


def d_e(t):
    return e(t + dt) - e(t)
    # return 15 * np.cos(t * 5 + 1)


def d_e2(t):
    return e(t + dt) * s(t + dt) - e(t) * s(t)


# def update_tracking(t):
#     global current_position

#     current_position = 2 * np.sin(2 * np.pi * 0.5 * t)


def get_tracking_target(t):
    # return 0
    return 2 * np.sin(2 * np.pi * 0.5 * t)


samples = []
e_ts = []
ts = []
t = 0
kp = 0.01
step = 0
while t < 8:
    samples.append(current_position)
    ts.append(t)
    e_ts.append(e(t))
    tracking_target = get_tracking_target(t)
    tracking_error = tracking_target - current_position

    target_position = tracking_error * kp + current_position + d_e2(t)
    # target_position = current_position + d_e(t)
    print(
        f"step : {step}, current_position: {current_position}, d_e(t): {d_e(t)}, e(t) : {e(t)}, tracking error * kp: {tracking_error * kp}"
    )
    # input()

    current_position = target_position
    print("==")

    t += dt
    step += 1
    # time.sleep(dt)

# plot samples
import matplotlib.pyplot as plt

plt.plot(ts, samples)
plt.plot(ts, e_ts)
plt.xlabel("Time (s)")
plt.ylabel("Position")
plt.title("Position over Time")
plt.grid()
plt.show()
