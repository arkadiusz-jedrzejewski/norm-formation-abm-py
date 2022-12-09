# Theoretical module
import numpy as np
import matplotlib.pyplot as plt


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


ps, cs = get_fixed_points_q_voter(num=100,
                                  q=3,
                                  f=0.5)

# plt.figure(1)
# plt.plot(ps, cs, '-')
#
# plt.show()
