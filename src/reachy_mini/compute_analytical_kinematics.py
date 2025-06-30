from sympy import *

# motor angle
q = symbols('q') 
# atachement point to the stewart platform
cx, cy, cz = symbols('cx cy cz')
# distance from the point to the stewart platform
banch_leg, motor_leg = symbols('bl ml')

joint_x = cos(q) * motor_leg
joint_y = sin(q) * motor_leg

# The point is at a distance rs from the point (px, py, pz)
radius_xy_plane = Eq((joint_x-cx)**2 + (joint_y-cy)**2 + (cz)**2, banch_leg**2)

res = solve(radius_xy_plane, q)

print("forward kinematics:")
print(res)

print("jacobian:")
print(simplify(Matrix([res[0]]).jacobian([cx, cy, cz])))
print(simplify(Matrix([res[1]]).jacobian([cx, cy, cz])))

print("compute the jacobian of the platform")


px, py, pz, proll, ppitch, pyaw = symbols('px py pz roll pitch yaw')

# Rotation matrices
Rx = Matrix([[1, 0, 0],
             [0, cos(proll), -sin(proll)],
             [0, sin(proll), cos(proll)]])

Ry = Matrix([[cos(ppitch), 0, sin(ppitch)],
             [0, 1, 0],
             [-sin(ppitch), 0, cos(ppitch)]])

Rz = Matrix([[cos(pyaw), -sin(pyaw), 0],
             [sin(pyaw), cos(pyaw), 0],
             [0, 0, 1]])

# Combined rotation: R = Rz * Ry * Rx (ZYX convention)
R = Rz * Ry * Rx



# Platform point in global frame
p = Matrix([px, py, pz])
ax, ay, az = symbols('ax ay az')  # attachment point in platform frame
bx, by, bz = symbols('bx by bz')  # base joint in global frame
a = Matrix([ax, ay, az])
b = Matrix([bx, by, bz])

platform_point = p + R * a
# Define rotation matrix R_b directly as a 3x3 symbolic matrix
T_b = MatrixSymbol('T_b', 4, 4)

leg_vector = T_b * Matrix([platform_point[0], platform_point[1], platform_point[2], 1])
#leg_vector = platform_point

print("leg_vector", simplify(leg_vector))

J = Matrix([leg_vector]).jacobian([px, py, pz, proll, ppitch, pyaw])
J = simplify(J)
print("jac_jacobian", J)