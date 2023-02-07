import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from theoretical_module import Logistic, Power, SymmetricPower, get_fixed_points


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
    x = x[0]
    ode_rhs = p * (nonconf_fun.get(x) - x) + (1 - p) * (
            (1 - x) * conf_fun.get(x) - x * conf_fun.get(1 - x))
    return ode_rhs


def model_ode_d(x, p, conf_fun, nonconf_fun):
    dF = p * (nonconf_fun.get_d(x) - 1) + (1 - p) * (
            (1 - x) * conf_fun.get_d(x) + x * conf_fun.get_d(1 - x) - conf_fun.get(x) - conf_fun.get(1 - x))
    return dF < 0


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


def model_quenched_ode_d(x, p, conf_fun, nonconf_fun):
    x2 = conf_fun.get(x) / (conf_fun.get(1 - x) + conf_fun.get(x))

    F11 = nonconf_fun.get_d(x) * p - 1
    F12 = nonconf_fun.get_d(x) * (1 - p)
    F21 = (1 - x2) * conf_fun.get_d(x) * p + x2 * conf_fun.get_d(1 - x) * p
    F22 = -conf_fun.get(x) + (1 - x2) * conf_fun.get_d(x) * (1 - p) \
          - conf_fun.get(1 - x) + x2 * conf_fun.get_d(1 - x) * (1 - p)

    det = F11 * F22 - F12 * F21
    tra = F11 + F22

    return np.logical_and(det > 0, tra < 0)


def diagram_fig(ax, col, num, conf_fun, nonconf_fun, is_quenched):
    p, x_fixed = get_fixed_points(num=num,
                                  conf_fun=conf_fun,
                                  nonconf_fun=nonconf_fun,
                                  is_quenched=is_quenched)
    ## stability checking
    if is_quenched:
        stable = model_quenched_ode_d(x=x_fixed,
                                      p=p,
                                      conf_fun=conf_fun,
                                      nonconf_fun=nonconf_fun)
    else:
        stable = model_ode_d(x=x_fixed,
                             p=p,
                             conf_fun=conf_fun,
                             nonconf_fun=nonconf_fun)
    x_stable = np.copy(x_fixed)
    x_stable[np.logical_not(stable)] = np.NAN

    x_unstable = np.copy(x_fixed)
    x_unstable[stable] = np.NAN

    ax.plot(p, x_stable, color=col)
    ax.plot(p, x_unstable, "--", color=col)

    return p, x_fixed


def diagram_ann_que_fig(fig, ax, num, conf_fun, nonconf_fun):
    fig.suptitle(nonconf_fun.__str__())
    p, _ = diagram_fig(ax[0],
                       col="r",
                       num=200,
                       conf_fun=conf_fun,
                       nonconf_fun=nonconf_fun,
                       is_quenched=False)
    p_max = max(p)
    p, _ = diagram_fig(ax[0],
                       col="b",
                       num=200,
                       conf_fun=conf_fun,
                       nonconf_fun=nonconf_fun,
                       is_quenched=True)
    p_max = max(p_max, max(p))

    ax[0].plot(4 * (q - 1) / (4 * q + k), 0.5,
               '.b')  # for SymmetricPower (=> annealed = quenched) / for Power quenched case
    ax[0].plot((q - 1) / (q - 1 + 2 ** (q - 1) * (k / 4 + 1)), 0.5, '.r')  # for Power annealed case
    ax[0].set_ylim([0, 1])
    ax[0].set_ylabel("$x^*$")
    ax[0].set_xlabel("$p$")
    ax[0].set_title(conf_fun.__str__())

    conf_fun = SymmetricPower(q=q)
    p, _ = diagram_fig(ax[1],
                col="r",
                num=200,
                conf_fun=conf_fun,
                nonconf_fun=nonconf_fun,
                is_quenched=False)
    p_max = max(p_max, max(p))

    diagram_fig(ax[1],
                col="b",
                num=200,
                conf_fun=conf_fun,
                nonconf_fun=nonconf_fun,
                is_quenched=True)
    ax[1].plot(4 * (q - 1) / (4 * q + k), 0.5,
               '.b')  # for SymmetricPower (=> annealed = quenched) / for Power quenched case
    ax[1].set_ylim([0, 1])
    ax[1].set_ylabel("$x^*$")
    ax[1].set_xlabel("$p$")
    ax[1].set_title(conf_fun.__str__())
    ax[0].set_xlim([0, p_max * 1.1])
    ax[1].set_xlim([0, p_max * 1.1])
    plt.tight_layout()

##
q = 3
conf_fun = Power(q=q)
x0, k, m = 0.5, 30, 0.5
nonconf_fun = Logistic(x0=x0, k=k, m=m)
##

fig, ax = plt.subplots(2, 1)
diagram_ann_que_fig(fig=fig,
                    ax=ax,
                    num=200,
                    conf_fun=conf_fun,
                    nonconf_fun=nonconf_fun)

plt.show()
p = 0.1
print(f"p: {p}")

t_max = 400
num = 400
t_eval = np.linspace(0, t_max, num)

plt.figure(1)
plt.plot([0, t_max], [0.5, 0.5], "k:")

for c_ini in np.linspace(0, 1, 40):
    print(c_ini)
    plt.figure(1)
    sol = solve_ivp(model_ode,
                    t_span=(0, t_max),
                    t_eval=t_eval,
                    y0=[c_ini],
                    args=(p, conf_fun, nonconf_fun),
                    method="LSODA")
    t, x = sol.t, sol.y[0, :]
    plt.plot(t, x, 'r')

    sol_que = solve_ivp(model_quenched_ode,
                        t_span=(0, t_max),
                        t_eval=t_eval,
                        y0=[c_ini, c_ini],
                        args=(p, conf_fun, nonconf_fun))
    t, x1, x2 = sol_que.t, sol_que.y[0, :], sol_que.y[1, :]
    x = x1 * p + x2 * (1 - p)
    plt.plot(t, x, 'b--')
    plt.figure(2)
    plt.plot(x1, x2, 'b')

plt.figure(1)
plt.title(f"p={p} " + conf_fun.__str__() + " " + nonconf_fun.__str__())
plt.xlim([0, t_max])
plt.ylim([0, 1])
plt.xlabel("MCS")
plt.ylabel("x")

plt.figure(2)
plt.title(f"p={p} " + conf_fun.__str__() + " " + nonconf_fun.__str__())
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel("x1")
plt.ylabel("x2")

x, y = np.meshgrid(np.linspace(0, 1, 10), np.linspace(0, 1, 10))
v_field = model_quenched_ode([], [x, y], p, conf_fun, nonconf_fun)
plt.quiver(x, y, v_field[0], v_field[1])

plt.show()
