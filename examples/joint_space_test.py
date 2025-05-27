from stewart_little_control import Dxl330IO
import time
import numpy as np

dxl_io = Dxl330IO("/dev/ttyACM0", baudrate=1000000, use_sync_read=True)
ids = [1, 2, 3, 4, 5, 6]
sign = [-1, 1, -1, 1, -1, 1]

dxl_io.enable_torque(ids)

def MoveTo(Angle_1, Angle_2, Angle_3, Angle_4, Angle_5, Angle_6, time_sleep):
        for i in range(1) :
                target = {}
                target[1] = Angle_1
                target[2] = Angle_2
                target[3] = Angle_3
                target[4] = Angle_4
                target[5] = Angle_5
                target[6] = Angle_6
                dxl_io.set_goal_position(target)
                time.sleep(time_sleep/10)
        return




MoveTo(-64,72,-10,10,-72,64, 10) # head In
MoveTo(-35,35,0,0,-35,35, 1) # Intermediate 2
MoveTo(-20,20,0,0,-20,20, 1) # Intermediate
MoveTo(0,0,0,0,0,0, 1) # 0


MoveTo(25,-25,25,-25,25,-25, 1) # Head Out
MoveTo(50,-50,50,-50,50,-50, 3) # Head Top
MoveTo(25,-25,25,-25,25,-25, 3) # Nominal

# Roll & Pitch 25°
MoveTo(41,-46,30,-20,10,-10, 3) # Roll -25°
MoveTo(25,-25,25,-25,25,-25, 1) # Nominal
MoveTo(10,-10,20,-30,46,-41, 3) # Roll +25°
MoveTo(25,-25,25,-25,25,-25, 1) # Nominal
MoveTo(39,-31,5,-5,31,-39, 3) # Pitch -25°
MoveTo(25,-25,25,-25,25,-25, 1) # Nominal
MoveTo(15,-15,45,-45,15,-15, 3) # Pitch +25°



for i in range(150) : # Yaw sinus 25°
    target = {}
    angle = (18*np.sin(2*np.pi*0.5*time.time()) )
    goal_pos = angle
    target[1] = goal_pos +25
    target[2] = goal_pos -25
    target[3] = goal_pos +25
    target[4] = goal_pos -25
    target[5] = goal_pos +25
    target[6] = goal_pos -25
    dxl_io.set_goal_position(target)
    time.sleep(0.01)

MoveTo(25,-25,25,-25,25,-25, 1) # Head Out

# Back & Forth / Left & Right Head
for j in range (2) :
        MoveTo(43,-16,29,-29,16,-43, 3) # Y -25mm
        MoveTo(15,-43,31,-31,43,-15, 3) # Y +25mm

for j in range (2) :
        MoveTo(38,-23,15,-46,39,-24, 3) # X -25mm
        MoveTo(24,-40,46,-15,22,-39, 3) # X +25mm

for j in range (2) :
        MoveTo(41,-61,50,-9,4,-22, 3) # X -25mm & Roll -25°
        MoveTo(25,-25,25,-25,25,-25, 1) # Head Out
        MoveTo(22,-5,9,-50,61,-41, 3) # X +25mm & Roll +25°
        MoveTo(25,-25,25,-25,25,-25, 1) # Head Out

for j in range (2) :
        MoveTo(41,-61,50,-9,4,-22, 1) # X -25mm & Roll -25°
        MoveTo(32,-68,48,-23,-5,-32, 3) # X -25mm & Roll -25° & Yaw -20°
        MoveTo(41,-61,50,-9,4,-22, 1) # X -25mm & Roll -25°
        MoveTo(52,-56,55,1,17,-18, 3) # X -25mm & Roll -25° & Yaw +20°

MoveTo(25,-25,25,-25,25,-25, 1) # Head Out
for j in range (2) :
        MoveTo(22,-5,9,-50,61,-41, 1) # X +25mm & Roll +25°
        MoveTo(29,3,20,-48,67,-34, 3) # X +25mm & Roll +25° & Yaw -20°
        MoveTo(22,-5,9,-50,61,-41, 1) # X +25mm & Roll +25°
        MoveTo(19,-16,1,-55,59,-52, 3) # X +25mm & Roll +25° & Yaw +20°


MoveTo(25,-25,25,-25,25,-25, 2) # Head Out
MoveTo(0,0,0,0,0,0, 3) # 0
MoveTo(-20,20,0,0,-20,20, 1) # Intermediate
MoveTo(-35,35,0,0,-35,35, 1) # Intermediate 2
MoveTo(-64,72,-10,10,-72,64, 100) # head In
