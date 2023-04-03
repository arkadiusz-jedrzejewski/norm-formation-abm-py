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

    def get(self, x):
        if hasattr(x, "__len__"):
            sol = np.copy(x)
            i = np.logical_and(1 >= x, x >= 0)
            sol[i] = 2.0 * self.m / (1 + np.exp(self.k * (sol[i] - self.x0)))
            sol[np.logical_not(i)] = 0
            return sol
        if 0 <= x <= 1:
            return 2.0 * self.m / (1 + np.exp(self.k * (x - self.x0)))
        elif x < 0:
            return 0
        elif x > 1:
            return 1

    def get_d(self, x):
        """
        returns derivative with respect to x evaluated at x
        :param x:
        :return:
        """
        if self.k == 0:
            return x * 0
        else:
            return - 2 * self.m * self.k * np.exp(self.k * (x - self.x0)) \
                / np.power(1 + np.exp(self.k * (x - self.x0)), 2)

    def __str__(self):
        return f"Logistic(x0={self.x0}, k={self.k:.4f}, m={self.m})"


class SymmetricPower:
    def __init__(self, q):
        self.q = q

    def get(self, x):
        if hasattr(x, "__len__"):
            sol = np.copy(x)
            i_less = np.logical_and(0.5 > x, x >= 0)
            i_more = np.logical_and(0.5 <= x, x <= 1)
            i_zero = np.logical_and(np.logical_not(i_less), np.logical_not(i_more))

            sol[i_less] = (2 * x[i_less]) ** self.q / 2
            sol[i_more] = 1 - (2 * (1 - x[i_more])) ** self.q / 2
            sol[i_zero] = 0
            return sol
        if 0.5 > x >= 0:
            return (2 * x) ** self.q / 2
        elif 0.5 <= x <= 1:
            return 1 - (2 * (1 - x)) ** self.q / 2
        else:
            return 0

    def get_d(self, x):
        """
        returns derivative with respect to x evaluated at x
        :param x:
        :return:
        """
        if self.q == 0:
            return x * 0
        else:
            if hasattr(x, "__len__"):
                sol = np.copy(x)
                i_less = np.logical_and(0.5 > x, x >= 0)
                i_more = np.logical_and(0.5 <= x, x <= 1)
                i_zero = np.logical_and(np.logical_not(i_less), np.logical_not(i_more))

                sol[i_less] = self.q * np.power(2 * x[i_less], self.q - 1)
                sol[i_more] = self.q * np.power(2 * (1 - x[i_more]), self.q - 1)
                sol[i_zero] = 0
                return sol
            if 0.5 > x >= 0:
                return self.q * np.power(2 * x, self.q - 1)
            elif 0.5 <= x <= 1:
                return self.q * np.power(2 * (1 - x), self.q - 1)
            else:
                return 0

    def __str__(self):
        return f"SymmetricPower(q={self.q})"


class Power:
    def __init__(self, q):
        self.q = q

    def get(self, x):
        return np.power(x, self.q)

    def get_d(self, x):
        """
        returns derivative with respect to x evaluated at x
        :param x:
        :return:
        """
        if self.q == 0:
            return x * 0
        else:
            return self.q * np.power(x, self.q - 1)

    def __str__(self):
        return f"Power(q={self.q})"


def get_fixed_points(num, conf_fun, nonconf_fun, is_quenched=False):
    """
    get_fixed_points returns the fixed points in the case of Bernoulli distribution
    :param num: number of points
    :param conf_fun: conformity function
    :param nonconf_fun: nonconformity function -> probability of engaging (opinion changes to 1)
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
        # (for other than Bernoulli distributions this results still holds
        # ps is the expected value of the distribution)
        numerator = cs * conf_fun.get(1 - cs) - (1 - cs) * conf_fun.get(cs)
        ps = numerator / (nonconf_fun.get(cs) - cs + numerator)

    return ps, cs


def quenched_fun(x, p, conf_fun, nonconf_fun):
    """
    function used to find numerically fixed points for the quenched model with Bernoulli distribution
    :param x:
    :param p:
    :param conf_fun:
    :param nonconf_fun:
    :return:
    """
    # print(x)
    numerator = x * (conf_fun.get(1 - x) + conf_fun.get(x)) - conf_fun.get(x)
    denominator = nonconf_fun.get(x) * (
            conf_fun.get(1 - x) + conf_fun.get(x)) - conf_fun.get(x)

    # print(numerator, denominator, p)
    return numerator / denominator - p


def model_quenched_ode_d(x, p, conf_fun, nonconf_fun):
    """
    function used to determine numerically the stability of fixed points for the quenched model with Bernoulli distribution
    :param x:
    :param p:
    :param conf_fun:
    :param nonconf_fun:
    :return:
    """
    x2 = conf_fun.get(x) / (conf_fun.get(1 - x) + conf_fun.get(x))

    F11 = nonconf_fun.get_d(x) * p - 1
    F12 = nonconf_fun.get_d(x) * (1 - p)
    F21 = (1 - x2) * conf_fun.get_d(x) * p + x2 * conf_fun.get_d(1 - x) * p
    F22 = -conf_fun.get(x) + (1 - x2) * conf_fun.get_d(x) * (1 - p) \
          - conf_fun.get(1 - x) + x2 * conf_fun.get_d(1 - x) * (1 - p)

    det = F11 * F22 - F12 * F21
    tra = F11 + F22

    return np.logical_and(det > 0, tra < 0)


def annealed_fun(x, p, conf_fun, nonconf_fun):
    """
    function used to find numerically fixed points for the annealed model (any distribution - p is the average)
    :param x:
    :param p:
    :param conf_fun:
    :param nonconf_fun:
    :return:
    """
    numerator = x * conf_fun.get(1 - x) - (1 - x) * conf_fun.get(x)
    return numerator / (nonconf_fun.get(x) - x + numerator) - p


def model_annealed_ode_d(x, p, conf_fun, nonconf_fun):
    """
    function used to determine numerically the stability of fixed points for the annealed model (any distribution - p is the average)
    :param x:
    :param p:
    :param conf_fun:
    :param nonconf_fun:
    :return:
    """
    dF = p * (nonconf_fun.get_d(x) - 1) + (1 - p) * (
            (1 - x) * conf_fun.get_d(x) + x * conf_fun.get_d(1 - x) - conf_fun.get(x) - conf_fun.get(1 - x))
    return dF < 0


def get_fixed_points_for(p, conf_fun, nonconf_fun, is_quenched):
    stable = []
    if is_quenched:
        # quenched model
        x_fixed = get_roots(lambda x: quenched_fun(x, p, conf_fun, nonconf_fun), 0, 1, 0.001)
        x_fixed.append(0.5)
        for x in x_fixed:
            stable.append(model_quenched_ode_d(x, p, conf_fun, nonconf_fun))
    else:
        # annealed model
        # (for other than Bernoulli distributions this results still holds
        # ps is the expected value of the distribution)
        x_fixed = get_roots(lambda x: annealed_fun(x, p, conf_fun, nonconf_fun), 0, 1, 0.001)
        x_fixed.append(0.5)
        for x in x_fixed:
            stable.append(model_annealed_ode_d(x, p, conf_fun, nonconf_fun))
    return x_fixed, stable


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


def c_p(p, c, conf_fun, nonconf_fun):
    """

    :param p:
    :param c:
    :param conf_fun: conformity function
    :param nonconf_fun: nonconformity function -> probability of engaging (opinion changes to 1)
    :return:
    """
    numerator = p * nonconf_fun.get(c) + (1 - p) * conf_fun.get(c)
    denominator = p + (1 - p) * (conf_fun.get(1 - c) + conf_fun.get(c))
    return numerator / denominator


def get_phase_diagram(p_start, p_stop, p_num, conf_fun, nonconf_fun, tol):
    ps = np.linspace(p_start, p_stop, p_num)
    cs = np.zeros(ps.shape)
    for i, p in enumerate(ps):
        c = fsolve(lambda x: x - integrate.quad(c_p, 0, p, (x, conf_fun, nonconf_fun))[0] / p,
                   cs[i - 1] if i > 0 else 1)
        cs[i] = c
        # error
        if np.abs(c - integrate.quad(c_p, 0, p, (c, conf_fun, nonconf_fun))[0] / p) > tol:
            cs[i] = np.NAN

        while np.isnan(cs[i]):
            c = fsolve(lambda x: x - integrate.quad(c_p, 0, p, (x, conf_fun, nonconf_fun))[0] / p,
                       np.random.rand())
            cs[i] = c
            if np.abs(c - integrate.quad(c_p, 0, p, (c, conf_fun, nonconf_fun))[0] / p) > tol:
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
            if x2 >= b:
                x2 = b
            # print("x1", x1, "roots", x2)
            f2 = f(x2)
            # print("f2", f2, "f1", f1)
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
                if root < b:
                    roots.append(root)
    return roots


def diff_p_symmetric_power_half(x, q, k):
    """
    first derivative of p_symmetric_power_half(x, q, k) with respect to x
    :param x: concentration of agents with opinion 1
    :param q: as in SymmetricPower(q) (considered only x<0.5)
    :param k: as in Logistic(x0=0.5, k, m=0.5)
    :return:
    """
    return (q * (2 * x) ** (q - 1) - 1) / ((2 * x) ** q / 2 - 1 / (np.exp(k * (x - 1 / 2)) + 1)) + (
            (q * (2 * x) ** (q - 1) + (k * np.exp(k * (x - 1 / 2))) / (np.exp(k * (x - 1 / 2)) + 1) ** 2) * (
            x - (2 * x) ** q / 2)) / ((2 * x) ** q / 2 - 1 / (np.exp(k * (x - 1 / 2)) + 1)) ** 2


def p_symmetric_power_half(x, q, k):
    """
    returns fixed point (p) in the case of quenched Bernoulli distribution (returns Bernoulli probability) or any
    annealed distribution (returns expected value of the distribution)

    Note: it is simplified version of get_fixed_points function where conf_fun=SymmetricPower(q) and
    nonconf_fun=Logistic(x0=0.5, k, m=0.5).
    The domain of this function x<0.5 (the diagram is symmetric along x=0.5 for these conditions)
    :param x: concentration of agents with opinion 1
    :param q: as in SymmetricPower(q) (considered only x<0.5)
    :param k: as in Logistic(x0=0.5, k, m=0.5)
    :return:
    """
    return (x - (2 * x) ** q / 2) / (1 / (1 + np.exp(k * (x - 0.5))) - (2 * x) ** q / 2)


def extrema_half(q, k, diff_p):
    """
    returns extrema of p_symmetric_power_half for x in (0, 0.5) and given parameters q and k
    :param q: as in SymmetricPower(q) (considered only x<0.5)
    :param k: as in Logistic(x0=0.5, k, m=0.5)
    :return:
    """
    sol = get_roots(lambda x: diff_p(x, q, k), 0, 0.5, 0.007)
    # print(sol)
    return sol


def ptd_symmetric_power_fun_k(q, ks):
    max_points = []
    min_points = []

    for k in ks:
        extrema = extrema_half(q, k, diff_p_symmetric_power_half)
        if len(extrema) == 2:
            extrema = [p_symmetric_power_half(x, q, k) for x in extrema]
            if extrema[0] > extrema[1]:
                max_points.append(extrema[0])
                min_points.append(extrema[1])
            else:
                max_points.append(extrema[1])
                min_points.append(extrema[0])
        else:
            max_points.append(None)
            min_points.append(None)
    return min_points, max_points


def ptd_symmetric_power_fun_q(qs, k):
    max_points = []
    min_points = []

    for q in qs:
        extrema = extrema_half(q, k, diff_p_symmetric_power_half)
        if len(extrema) == 2:
            extrema = [p_symmetric_power_half(x, q, k) for x in extrema]
            if extrema[0] > extrema[1]:
                max_points.append(extrema[0])
                min_points.append(extrema[1])
            else:
                max_points.append(extrema[1])
                min_points.append(extrema[0])
        else:
            max_points.append(None)
            min_points.append(None)
    return min_points, max_points


def diff_p_power(x, q, k, is_quenched):
    if is_quenched:
        return - ((1 - x) ** q + x ** q - q * x ** (q - 1) - x * (q * (1 - x) ** (q - 1) - q * x ** (q - 1))) / (
                x ** q - ((1 - x) ** q + x ** q) / (np.exp(k * (x - 1 / 2)) + 1)) - (
                (x ** q - x * ((1 - x) ** q + x ** q)) * (
                (q * (1 - x) ** (q - 1) - q * x ** (q - 1)) / (np.exp(k * (x - 1 / 2)) + 1) + q * x ** (q - 1) + (
                k * np.exp(k * (x - 1 / 2)) * ((1 - x) ** q + x ** q)) / (np.exp(k * (x - 1 / 2)) + 1) ** 2)) / (
                x ** q - ((1 - x) ** q + x ** q) / (np.exp(k * (x - 1 / 2)) + 1)) ** 2
    else:
        return - ((1 - x) ** q + x ** q - q * x ** (q - 1) - x * (q * (1 - x) ** (q - 1) - q * x ** (q - 1))) / (
                x + x ** q - x * ((1 - x) ** q + x ** q) - 1 / (np.exp(k * (x - 1 / 2)) + 1)) - (
                (x ** q - x * ((1 - x) ** q + x ** q)) * (
                q * x ** (q - 1) - x ** q - (1 - x) ** q + x * (q * (1 - x) ** (q - 1) - q * x ** (q - 1)) + (
                k * np.exp(k * (x - 1 / 2))) / (np.exp(k * (x - 1 / 2)) + 1) ** 2 + 1)) / (
                x + x ** q - x * ((1 - x) ** q + x ** q) - 1 / (np.exp(k * (x - 1 / 2)) + 1)) ** 2


def p_power(x, q, k, is_quenched):
    if is_quenched:
        return (x ** q - x * ((1 - x) ** q + x ** q)) / (
                x ** q - ((1 - x) ** q + x ** q) / (np.exp(k * (x - 1 / 2)) + 1))
    else:
        return (x ** q - x * ((1 - x) ** q + x ** q)) / (
                x + x ** q - x * ((1 - x) ** q + x ** q) - 1 / (np.exp(k * (x - 1 / 2)) + 1))


def ptd_power_fun_k(q, ks, is_quenched):
    points = []
    ks_res = []

    for k in ks:
        extrema = extrema_half(q, k, lambda x, y, z: diff_p_power(x, y, z, is_quenched))
        if len(extrema) != 0:
            for item in extrema:
                points.append(p_power(item, q, k, is_quenched))
                ks_res.append(k)
    return points, ks_res


def ptd_power_fun_q(qs, k, is_quenched):
    points = []
    qs_res = []

    for q in qs:
        extrema = extrema_half(q, k, lambda x, y, z: diff_p_power(x, y, z, is_quenched))
        if len(extrema) != 0:
            for item in extrema:
                points.append(p_power(item, q, k, is_quenched))
                qs_res.append(q)
    return points, qs_res


def plot_diagram_symmetric_power_fun_k(ax, q, ks):
    min_points, max_points = ptd_symmetric_power_fun_k(q, ks)
    # plt.figure()

    min_points_nf = []  # nonefree
    max_points_nf = []
    ks_nf = []
    for i, item in enumerate(min_points):
        if item is not None:
            min_points_nf.append(item)
            max_points_nf.append(max_points[i])
            ks_nf.append(ks[i])

    ax.plot3D(ks_nf, min_points_nf, [0.5] * len(ks_nf))
    ax.plot3D(ks_nf, max_points_nf, [0.5] * len(ks_nf))
    ax.plot3D(ks, 4 * (q - 1) / (4 * q + ks), [0.5] * len(ks))
    # plt.title(f"SymmetricPower(q={q}) Logistic(x0=0.5, k, m=0.5) any distribution\nannealed=quenched")
    # plt.xlabel("k")
    # plt.ylabel("p")


def plot_diagram_symmetric_power_fun_q(qs, k):
    min_points, max_points = ptd_symmetric_power_fun_q(qs, k)
    plt.figure()
    plt.plot(qs, min_points)
    plt.plot(qs, max_points)
    plt.plot(qs, 4 * (qs - 1) / (4 * qs + k))
    plt.title(f"SymmetricPower(q) Logistic(x0=0.5, k={k}, m=0.5) any distribution\nannealed=quenched")
    plt.xlabel("q")
    plt.ylabel("p")


def plot_diagram_symmetric_power(qs, ks):
    q_sol = []
    k_sol = []
    for q in qs:
        min_points, max_points = ptd_symmetric_power_fun_k(q, ks)
        sol = [False if x is None else True for x in min_points]
        q_sol.append(q)
        k_sol.append(min(ks[sol]))

    plt.figure()
    plt.plot(q_sol, k_sol)
    np.savetxt("test2.txt", np.column_stack((q_sol, k_sol)))
    plt.xlabel("q")
    plt.ylabel("k")
    # plt.xlim([1, 8])
    plt.ylim([20, 40])
    plt.title("SymmetricPower any distribution annealed=quenched")
    print(len(q_sol))


def plot_diagram_power_fun_k(axs, q, ks, is_quenched):
    ps, ks_p = ptd_power_fun_k(q, ks, is_quenched)
    # plt.figure()
    # ax = plt.axes(projection='3d')
    axs.plot3D(ks_p, ps, [0.5] * len(ps))
    if is_quenched:
        axs.plot3D(ks, 4 * (q - 1) / (4 * q + ks), [0.5] * len(ks), 'r')  # for Power quenched case
        axs.plot3D(ks_p, 4 * (q - 1) / (4 * q + np.array(ks_p)), [0.5] * len(ks_p), 'm')
        # plt.title(f"Power(q={q}) Logistic(x0=0.5, k, m=0.5) Bernoulli quenched")
    else:
        axs.plot3D(ks, (q - 1) / (q - 1 + 2 ** (q - 1) * (ks / 4 + 1)), [0.5] * len(ks), 'r')  # for Power annealed case
        # plt.title(f"Power(q={q}) Logistic(x0=0.5, k, m=0.5) Bernoulli annealed")
    # plt.xlabel("k")
    # plt.ylabel("p")


def plot_diagram_power_fun_q(qs, k, is_quenched):
    ps, qs_p = ptd_power_fun_q(qs, k, is_quenched)
    plt.figure()
    plt.plot(qs_p, ps)
    if is_quenched:
        plt.plot(qs, 4 * (qs - 1) / (4 * qs + k), 'r')  # for Power quenched case
        plt.title(f"Power(q) Logistic(x0=0.5, k={k}, m=0.5) Bernoulli quenched")
    else:
        plt.plot(qs, (qs - 1) / (qs - 1 + 2 ** (qs - 1) * (k / 4 + 1)), 'r')  # for Power annealed case
        plt.title(f"Power(q) Logistic(x0=0.5, k={k}, m=0.5) Bernoulli annealed")
    plt.xlabel("q")
    plt.ylabel("p")


def plot_diagram_power(q, k, is_quenched):
    plt.figure()
    if is_quenched:
        q = -(2 * k - ((k + 2) * (k + 4) * (k ** 2 - 2 * k + 8)) ** (1 / 2) + 8) / (4 * (k + 4))
        plt.title("Power quenched Bernoulli distribution")
    else:
        k = ((40 * q) / 3 - (8 * q ** 2) / 3) / (
                80 * q + ((80 * q - 16 * q ** 2) ** 2 - ((40 * q) / 3 - (8 * q ** 2) / 3) ** 3) ** (
                1 / 2) - 16 * q ** 2) ** (1 / 3) + (
                    80 * q + ((- 16 * q ** 2 + 80 * q) ** 2 - ((40 * q) / 3 - (8 * q ** 2) / 3) ** 3) ** (
                    1 / 2) - 16 * q ** 2) ** (1 / 3)
        plt.title("Power annealed Bernoulli distribution")
    plt.plot(q, k)
    plt.xlabel("q")
    plt.ylabel("k")
    plt.xlim([1, 8])

# qs = np.linspace(1.001, 8, 15)
# k = np.linspace(0, 40, 100)
# print(k[1] - k[0])
# solq = []
# plt.figure(100)
# for q in qs:
#     sol = []
#     for ki in k:
#         soli = [p_symmetric_power_half(x, q, ki) for x in extrema_symmetric_power_half(q, ki)]
#         sol.append(soli)
#
#         # plt.plot([ki] * len(soli), soli, '.b')
#     # plt.plot(k, 4 * (q - 1) / (4 * q + k),
#     #          'k')  # for SymmetricPower (=> annealed = quenched) / for Power quenched case
#     # plt.xlabel('k')
#     # plt.ylabel('p')
#     # plt.show()
#     print(sol)
#     sol = [len(x) != 0 for x in sol]
#     #print(sol)
#     ind = sol.index(True)
#     print(q, k[ind])
#     solq.append(k[ind])
# plt.plot(qs, solq, ':.')
# plt.xlabel("q")
# plt.ylabel("k")
# plt.title("SymmetricPower Bernoulli distribution")
# print("lol", extrema_symmetric_power_half(2.500785714285714, 0.40404040404040403))
# plt.plot(conc, nonf.get(conc))
# plt.ylim((0, 1))
# plt.show()
# q = 2
# x0 = 0.5
# k = 15
# m = 0.6
# ps = np.linspace(0.001, 1.999, 100)
#
# conf_fun = Power(q=q)
# print(conf_fun)
#
# nonconf_fun = Logistic(x0=x0, k=k, m=m)
# print(nonconf_fun)
#
# c, a = np.mgrid[0:1:0.001, 0:1:0.001]
# z = (conf_fun.get(c) - nonconf_fun.get(c)) / (conf_fun.get(1 - c) + conf_fun.get(c) - 1) - c + (
#             conf_fun.get(c) - nonconf_fun.get(c) * (conf_fun.get(1 - c) + conf_fun.get(c))) / (
#                 a * (conf_fun.get(1 - c) + conf_fun.get(c) - 1) ** 2) * np.log(
#     1 - a + a / (conf_fun.get(c) + conf_fun.get(1 - c)))
# plt.contour(1 - a/2, c, z, 0, colors='k')


# x=0.2
# print("addd: ",conf_fun.get(x) + conf_fun.get(1 - x))
# roots = []
# for p in ps:
#     f1 = lambda x: x - integrate.quad(c_p, 0, p, (x, conf_fun, nonconf_fun))[0] / p
#     # f1 = lambda x: x - p * nonconf_fun.get(x) - (1 - p) * conf_fun.get(x) #/ (conf_fun.get(x) + conf_fun.get(1 - x))
#     root = get_roots(f1, 0, 1, 0.0011)
#     print(f"p:\t{p}\t{root}")
#     #print(p)
#     plt.plot(1 - p * np.ones(len(root))/2, root, '.r')
#     roots = roots + root
#
# plt.xlim([0, 1])
# plt.ylim([0, 1])
# plt.title(conf_fun.__str__() + " " + nonconf_fun.__str__())
# plt.xlabel("nonconformity")
# plt.ylabel("concentration")
#
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
