import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from theoretical_module import Logistic, Power, SymmetricPower


def conformity_function_ode(x, q) -> float:
    """
    Returns the probability of conforming to the group pressure
        Parameters:
            x (float): a fraction of adopters
            q (float): a nonlinearity parameter, q>=0

        Returns:
            f (float): probability of conforming to the group pressure
    """
    # f = np.power(x, q)
    if x < 0.5:
        f = np.power(2 * x, q) / 2
    else:
        f = 1 - np.power(2 * (1 - x), q) / 2
    return f


def engagement_probability(x, x0, k, m=0.5) -> float:
    """
    Returns the probability of engaging in a given bahavior
        Parameters:
            x (float): a fraction of adopters
            x0 (float): a location parameter, x0 in [0, 1]
            k (float): a scale parameter

        Returns:
            f (float): probability of engaging in a given bahavior
    """
    f = 2 * m / (1 + np.exp(k * (x - x0)))
    return f


def model_ode(t, x, p, conf_fun, nonconf_fun):
    """
    annealed version - Bernoulli distribution
    :param t:
    :param x:
    :param p:
    :param q:
    :param x0:
    :param k:
    :param m:
    :return:
    """
    ode_rhs = p * (nonconf_fun.get(x) - x) + (1 - p) * (
            (1 - x) * conf_fun.get(x) - x * conf_fun.get(1 - x))
    return ode_rhs


def model_quenched_ode(t, xs, p, conf_fun, nonconf_fun):
    """

    :param t:
    :param xs:
    :param p:
    :param conf_fun:
    :param nonconf_fun:
    :return:
    """
    x = xs[0] * p + xs[1] * (1 - p)

    ode_rhs = [nonconf_fun.get(x) - xs[0],
               (1 - xs[1]) * conf_fun.get(x) - xs[1] * conf_fun.get(1 - x)
               ]
    return ode_rhs


p = 0.25

q = 2
conf_fun = Power(q=q)
x0, k, m = 0.5, 0, 0.5
nonconf_fun = Logistic(x0=x0, k=k, m=m)

t_max = 50
num = 100
t_eval = np.linspace(0, t_max, num)
plt.plot([0, t_max], [0.5, 0.5], "k:")


for c_ini in np.linspace(0, 1, 13):
    plt.figure(1)
    sol = solve_ivp(model_ode,
                    t_span=(0, t_max),
                    t_eval=t_eval,
                    y0=[c_ini],
                    args=(p, conf_fun, nonconf_fun))
    t, x = sol.t, sol.y[0, :]
    plt.plot(t, x, 'r')

    sol_que = solve_ivp(model_quenched_ode,
                        t_span=(0, t_max),
                        t_eval=t_eval,
                        y0=[c_ini, c_ini],
                        args=(p, conf_fun, nonconf_fun))
    t, x1, x2 = sol_que.t, sol_que.y[0, :], sol_que.y[1, :]
    x = x2 * p + x2 * (1 - p)
    plt.plot(t, x, 'b')
    plt.figure(2)
    plt.plot(x1, x2)


plt.figure(1)
plt.xlim([0, t_max])
plt.ylim([0, 1])
plt.xlabel("MCS")
plt.ylabel("x")

plt.figure(2)
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()
