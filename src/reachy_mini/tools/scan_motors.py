from rustypot import Xl330PyController

baudrates = [9600, 57600, 115200, 1000000]


def scan(baudrate):
    c = Xl330PyController("/dev/ttyAMA3", baudrate, 0.01)
    found_motors = []
    for i in range(255):
        ret = c.ping(i)
        if ret:
            found_motors.append(i)
    return found_motors


for baudrate in baudrates:
    print(f"Trying baudrate: {baudrate}")
    found_motors = scan(baudrate)
    if found_motors:
        print(f"Found motors at baudrate {baudrate}: {found_motors}")
    else:
        print(f"No motors found at baudrate {baudrate}")
