import numpy as np
from numpy import pi
from scipy.optimize import fsolve

import uwf_model

S = uwf_model.S

a = 2.2e-3
h = 100e-9
kappa = 2*pi/10e-6

r = uwf_model.r(h, kappa, a)

def to_solve(q):
    return q * np.imag(S(q)) + 1/r

zero = fsolve(to_solve, 0.4)

