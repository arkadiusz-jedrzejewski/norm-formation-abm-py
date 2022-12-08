# Theoretical module
import numpy as np
import matplotlib.pyplot as plt


def conformity_function(x, q):
    return x ** q


def nonconformity_function(x, f):
    return f


def get_fixed_points(num, q, f):
    cs = np.linspace(0, 1, num=num)
    # quenched model
    numerator = cs * (conformity_function(1 - cs, q) + conformity_function(cs, q)) - conformity_function(cs, q)
    denominator = nonconformity_function(cs, f) * (
            conformity_function(1 - cs, q) + conformity_function(cs, q)) - conformity_function(cs, q)
    ps = numerator / denominator

    return ps, cs


ps, cs = get_fixed_points(num=100,
                          q=3,
                          f=0.5)

plt.figure(1)
plt.plot(ps, cs, '-')

plt.show()
