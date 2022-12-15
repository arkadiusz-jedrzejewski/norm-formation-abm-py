# Theoretical module
import numpy as np
from scipy.special import lambertw


def conformity_function(x, q):
    return x ** q


def nonconformity_function(x, f):
    return f


def get_fixed_points(num, q, f, is_quenched=False):
    cs = np.linspace(0, 1, num=num)
    if is_quenched:
        # quenched model
        numerator = cs * (conformity_function(1 - cs, q) + conformity_function(cs, q)) - conformity_function(cs, q)
        denominator = nonconformity_function(cs, f) * (
                conformity_function(1 - cs, q) + conformity_function(cs, q)) - conformity_function(cs, q)
        ps = numerator / denominator
    else:
        numerator = cs * conformity_function(1 - cs, q) - (1 - cs) * conformity_function(cs, q)
        ps = numerator / (nonconformity_function(cs, f) - cs + numerator)

    return ps, cs


def get_fixed_points_uniform(num, q, f, is_quenched=False):
    cs = np.linspace(0.001, 0.999, num=num)
    a1 = cs - (conformity_function(cs, q) - nonconformity_function(cs, f)) / (conformity_function(1 - cs, q) + conformity_function(cs, q) - 1)
    a2 = (conformity_function(cs, q) - nonconformity_function(cs, f)*(conformity_function(1-cs, q)+conformity_function(cs, q)))/(conformity_function(1-cs, q)+conformity_function(cs, q)-1)**2
    a3 = conformity_function(cs, q) + conformity_function(1-cs, q)
    if is_quenched:
        # quenched model
        ps = ((1-a3)*a2*lambertw(a1*a3*np.exp(a1*a3/a2/(a3-1))/a2/(a3-1))+a1*a3)/(a1*(a3-1))
    else:
        numerator = cs * conformity_function(1 - cs, q) - (1 - cs) * conformity_function(cs, q)
        ps = numerator / (nonconformity_function(cs, f) - cs + numerator)

    return ps, cs


def get_fixed_points_q_voter(num, q, f, is_quenched=False):
    cs = np.linspace(0, 1, num=num)
    if is_quenched:
        # quenched model
        numerator = cs * (1 - cs) ** q - (1 - cs) * cs ** q
        ps = 2 * numerator / ((1 - cs) ** q - cs ** q)
    else:
        # annealed model
        numerator = cs * (1 - cs) ** q - (1 - cs) * cs ** q
        ps = numerator / ((1 - 2 * cs) * f + numerator)

    return ps, cs
