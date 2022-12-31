# Theoretical module
import math

import numpy as np
from scipy.special import lambertw
import scipy.integrate as integrate
from scipy.optimize import minimize, Bounds, fsolve

import matplotlib.pyplot as plt


def conformity_function(x, q):
    return x ** q


def nonconformity_function(x, x0, k, m):
    return 2.0 * m / (1 + np.exp(k * (x - x0)))


# def get_fixed_points(num, q, f, is_quenched=False):
#     cs = np.linspace(0, 1, num=num)
#     if is_quenched:
#         # quenched model
#         numerator = cs * (conformity_function(1 - cs, q) + conformity_function(cs, q)) - conformity_function(cs, q)
#         denominator = nonconformity_function(cs, f) * (
#                 conformity_function(1 - cs, q) + conformity_function(cs, q)) - conformity_function(cs, q)
#         ps = numerator / denominator
#     else:
#         numerator = cs * conformity_function(1 - cs, q) - (1 - cs) * conformity_function(cs, q)
#         ps = numerator / (nonconformity_function(cs, f) - cs + numerator)
#
#     return ps, cs


# def get_fixed_points_uniform(num, q, f, is_quenched=False):
#     cs = np.linspace(0.001, 0.999, num=num)
#     a1 = cs - (conformity_function(cs, q) - nonconformity_function(cs, f)) / (
#             conformity_function(1 - cs, q) + conformity_function(cs, q) - 1)
#     a2 = (conformity_function(cs, q) - nonconformity_function(cs, f) * (
#             conformity_function(1 - cs, q) + conformity_function(cs, q))) / (
#                  conformity_function(1 - cs, q) + conformity_function(cs, q) - 1) ** 2
#     a3 = conformity_function(cs, q) + conformity_function(1 - cs, q)
#     if is_quenched:
#         # quenched model
#         ps = ((1 - a3) * a2 * lambertw(a1 * a3 * np.exp(a1 * a3 / a2 / (a3 - 1)) / a2 / (a3 - 1)) + a1 * a3) / (
#                 a1 * (a3 - 1))
#     else:
#         numerator = cs * conformity_function(1 - cs, q) - (1 - cs) * conformity_function(cs, q)
#         ps = numerator / (nonconformity_function(cs, f) - cs + numerator)
#
#     return ps, cs


# def get_fixed_points_q_voter(num, q, f, is_quenched=False):
#     cs = np.linspace(0, 1, num=num)
#     if is_quenched:
#         # quenched model
#         numerator = cs * (1 - cs) ** q - (1 - cs) * cs ** q
#         ps = 2 * numerator / ((1 - cs) ** q - cs ** q)
#     else:
#         # annealed model
#         numerator = cs * (1 - cs) ** q - (1 - cs) * cs ** q
#         ps = numerator / ((1 - 2 * cs) * f + numerator)
#
#     return ps, cs


def fun(x, c, q, x0, k, m):
    # nonconformity_function -> probability of engaging (opinion changes to 1)
    numerator = x * nonconformity_function(c, x0, k, m) + (1 - x) * conformity_function(c, q)
    denominator = x + (1 - x) * (conformity_function(1 - c, q) + conformity_function(c, q))
    return numerator / denominator


def get_phase_diagram(p_start, p_stop, p_num, q, x0, k, m, tol):
    ps = np.linspace(p_start, p_stop, p_num)
    cs = np.zeros(ps.shape)
    for i, p in enumerate(ps):
        c = fsolve(lambda x: x - integrate.quad(fun, 0, p, (x, q, x0, k, m))[0] / p,
                   cs[i - 1] if i > 0 else 1)
        cs[i] = c
        # error
        if np.abs(c - integrate.quad(fun, 0, p, (c, q, x0, k, m))[0] / p) > tol:
            cs[i] = np.NAN

        while np.isnan(cs[i]):
            c = fsolve(lambda x: x - integrate.quad(fun, 0, p, (x, q, x0, k, m))[0] / p,
                       np.random.rand())
            cs[i] = c
            if np.abs(c - integrate.quad(fun, 0, p, (c, q, x0, k, m))[0] / p) > tol:
                cs[i] = np.NAN

    return ps, cs


def rootsearch(f, a, b, dx) -> tuple:
    """
    For function f(x), rootsearch searches interval (a, b) in increments dx, and it finds interval (x1, x2)
    that contains the smallest root of f(x).
    It returns (None, None) if no roots were detected.

    Numerical methods in engineering with python 3
    Chapter 4.2

    :param f:   function
    :param a:   lower search bound
    :param b:   upper search bound
    :param dx:  increment
    :return:    (x1, x2): an interval that contains the smallest root of f(x)
    """
    x1, f1 = a, f(a)
    x2 = a + dx
    f2 = f(x2)
    while f1 * f2 > 0:
        if x1 >= b:
            return None, None
        else:
            x1, f1 = x2, f2
            x2 = x1 + dx
            f2 = f(x2)
    else:
        return x1, x2


def bisection(f, x1, x2, switch=False, tol=1e-9) -> float:
    """
    Bisection finds a root of f(x) = 0 by bisection.
    The root must be inside interval (x1, x2).
    Setting switch = True returns root = None if f(x) increases with bisection

    :param f:       function
    :param x1:      lower bound
    :param x2:      upper bound
    :param switch:  switch = 1 returns root = None if f(x) increases with bisection
    :param tol:     error tolerance
    :return:        estimated root of f(x) = 0
    """
    f1 = f(x1)
    if f1 == 0:
        return x1
    f2 = f(x2)
    if f2 == 0:
        return x2
    if f1 * f2 > 0:
        return None
    n = int(math.ceil(math.log(abs(x2 - x1) / tol) / math.log(2)))

    for i in range(n):
        x3 = (x1 + x2) / 2
        f3 = f(x3)
        if switch and abs(f3) > abs(f1) and abs(f3) > abs(f2):
            return None
        if f3 == 0:
            return x3
        if f2 * f3 < 0:
            x1, f1 = x3, f3
        else:
            x2, f2 = x3, f3

    return (x1 + x2) / 2


def get_roots(f, a, b, dx) -> list:
    """
    get_roots finds roots of f(x) = 0 in interval (a, b)

    :param f:   function
    :param a:   lower bound
    :param b:   upper bound
    :param dx:  increment
    :return:    list of found roots of f(x) = 0
    """
    roots = []
    while True:
        x1, x2 = rootsearch(f, a, b, dx)
        if x1 is None:
            break
        else:
            a = x2
            root = bisection(f, x1, x2, True)
            if root is not None:
                roots.append(root)
    return roots


q = 1.5
x0 = 0.5
k = 0
m = 0.6
ps = np.linspace(0.001, 0.999, 200)
roots = []
for p in ps:
    f1 = lambda x: x - integrate.quad(fun, 0, p, (x, q, x0, k, m))[0] / p
    root = get_roots(f1, 0, 1, 0.001)
    #print(f"p:\t{p}\t{root}")
    plt.plot(p * np.ones(len(root)), root, '.c')
    roots = roots + root
plt.show()
# ps, cs = get_phase_diagram(p_start=0,
#                            p_stop=1,
#                            p_num=110,
#                            q=5,
#                            f=0.45,
#                            tol=1e-6)
#
# print(ps, cs, sep='\n')
# plt.plot(ps, cs, '.-')
# plt.ylim(0, 1)
# plt.show()
