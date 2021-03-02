import numpy as np
def calR(ω,ϕ,κ):
    from math import *
    return np.asarray([cos(ϕ)*cos(κ),
    -cos(ϕ)*sin(κ),
    sin(ϕ),
    cos(ω)*sin(κ)+sin(ω)*sin(ϕ)*cos(κ),
    cos(ω)*cos(κ)-sin(ω)*sin(ϕ)*sin(κ),
    -sin(ω)*cos(ϕ),
    sin(ω)*sin(κ)-cos(ω)*sin(ϕ)*cos(κ),
    sin(ω)*cos(κ)+cos(ω)*sin(ϕ)*sin(κ),
    cos(ω)*cos(ϕ)]).reshape([3,3])
