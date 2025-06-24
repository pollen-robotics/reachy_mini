from sympy import *

alpha = symbols('alpha')
px, py, pz = symbols('px py pz')
rs, rp = symbols('rs rp')

ex = cos(alpha) * rs
ey = sin(alpha) * rs

# The point is at a distance rs from the point (px, py, pz)
e1 = Eq((ex-px)**2 + (ey-py)**2 + (pz)**2, rp**2)

r = solve(e1, alpha)

print(r)