# Theoretical module
import math

import numpy as np
from scipy.special import lambertw
import scipy.integrate as integrate
from scipy.optimize import minimize, Bounds, fsolve

import matplotlib.pyplot as plt


class Logistic:
    def __init__(self, x0, k, m):
        self.x0 = x0
        self.k = k
        self.m = m

    def get(self, conc):
        return 2.0 * self.m / (1 + np.exp(self.k * (conc - self.x0)))


class SymmetricPower:
    def __init__(self, q):
        self.q = q

    def get(self, conc):
        if hasattr(conc, "__len__"):
            result = []
            for c in conc:
                if c < 0.5:
                    result.append(np.power(2 * c, self.q) / 2)
                else:
                    result.append(1 - np.power(2 * (1 - c), self.q) / 2)
            return np.array(result)
        if conc < 0.5:
            return np.power(2 * conc, self.q) / 2
        else:
            return 1 - np.power(2 * (1 - conc), self.q) / 2


def get_fixed_points(num, conf_fun, nonconf_fun, is_quenched=False):
    """
    get_fixed_points returns the fixed points in the case of Bernoulli distribution
    :param num: number of points
    :param conf_fun: conformity function
    :param nonconf_fun: nonconformity function
    :param is_quenched:
    :return:
    """
    cs = np.linspace(0, 1, num=num)
    if is_quenched:
        # quenched model
        numerator = cs * (conf_fun.get(1 - cs) + conf_fun.get(cs)) - conf_fun.get(cs)
        denominator = nonconf_fun.get(cs) * (
                conf_fun.get(1 - cs) + conf_fun.get(cs)) - conf_fun.get(cs)
        ps = numerator / denominator
    else:
        # annealed model
        numerator = cs * conf_fun.get(1 - cs) - (1 - cs) * conf_fun.get(cs)
        ps = numerator / (nonconf_fun.get(cs) - cs + numerator)

    return ps, cs


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


# def fun(x, c, q, x0, k, m):
#     # nonconformity_function -> probability of engaging (opinion changes to 1)
#     numerator = x * nonconformity_function(c, x0, k, m) + (1 - x) * conformity_function(c, q)
#     denominator = x + (1 - x) * (conformity_function(1 - c, q) + conformity_function(c, q))
#     return numerator / denominator


# def get_phase_diagram(p_start, p_stop, p_num, q, x0, k, m, tol):
#     ps = np.linspace(p_start, p_stop, p_num)
#     cs = np.zeros(ps.shape)
#     for i, p in enumerate(ps):
#         c = fsolve(lambda x: x - integrate.quad(fun, 0, p, (x, q, x0, k, m))[0] / p,
#                    cs[i - 1] if i > 0 else 1)
#         cs[i] = c
#         # error
#         if np.abs(c - integrate.quad(fun, 0, p, (c, q, x0, k, m))[0] / p) > tol:
#             cs[i] = np.NAN
#
#         while np.isnan(cs[i]):
#             c = fsolve(lambda x: x - integrate.quad(fun, 0, p, (x, q, x0, k, m))[0] / p,
#                        np.random.rand())
#             cs[i] = c
#             if np.abs(c - integrate.quad(fun, 0, p, (c, q, x0, k, m))[0] / p) > tol:
#                 cs[i] = np.NAN
#
#     return ps, cs


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


conf_fun = SymmetricPower(1.5)
nonconf_fun = Logistic(0.5, 0, 0.5)

ps, cs = get_fixed_points(100, conf_fun, nonconf_fun, is_quenched=True)
plt.plot(ps, cs)
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel("nonconformity probability")
plt.ylabel("concentration")
plt.show()
# plt.plot(conc, nonf.get(conc))
# plt.ylim((0, 1))
# plt.show()
# q = 1.5
# x0 = 0.5
# k = 0
# m = 0.6
# ps = np.linspace(0.001, 0.999, 200)
# roots = []
# for p in ps:
#     f1 = lambda x: x - integrate.quad(fun, 0, p, (x, q, x0, k, m))[0] / p
#     root = get_roots(f1, 0, 1, 0.001)
#     #print(f"p:\t{p}\t{root}")
#     plt.plot(p * np.ones(len(root)), root, '.c')
#     roots = roots + root
# plt.show()
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
