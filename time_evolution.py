import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from theoretical_module import Logistic, Power, SymmetricPower, get_fixed_points, get_roots


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
    ax.plot([0, 1], [0.5, 0.5], ":k")
    p, _ = diagram_fig(ax,
                       col="r",
                       num=num,
                       conf_fun=conf_fun,
                       nonconf_fun=nonconf_fun,
                       is_quenched=False)
    p_max = max(p)
    p, _ = diagram_fig(ax,
                       col="b",
                       num=num,
                       conf_fun=conf_fun,
                       nonconf_fun=nonconf_fun,
                       is_quenched=False)
    p_max = max(p_max, max(p))

    ax.set_ylim([0, 1])
    ax.set_ylabel("$x^*$")
    ax.set_xlabel("$p$")
    ax.set_title(conf_fun.__str__())

    ax.set_xlim([0, min(p_max * 1.1, 1)])
    return p_max


def get_time_evolution(p, t_max, num, c_ini, conf_fun, nonconf_fun, is_quenched):
    t_eval = np.linspace(0, t_max, num)
    if is_quenched:
        sol = solve_ivp(model_quenched_ode,
                        t_span=(0, t_max),
                        t_eval=t_eval,
                        y0=c_ini,
                        args=(p, conf_fun, nonconf_fun))
        t, x1, x2 = sol.t, sol.y[0, :], sol.y[1, :]
        x = x1 * p + x2 * (1 - p)
        return t, x1, x2, x
    else:
        sol = solve_ivp(model_ode,
                        t_span=(0, t_max),
                        t_eval=t_eval,
                        y0=c_ini,
                        args=(p, conf_fun, nonconf_fun),
                        method="LSODA")
        t, x = sol.t, sol.y[0, :]
        return t, x


def quenched_fun(x, p, conf_fun, nonconf_fun):
    # print(x)
    numerator = x * (conf_fun.get(1 - x) + conf_fun.get(x)) - conf_fun.get(x)
    denominator = nonconf_fun.get(x) * (
            conf_fun.get(1 - x) + conf_fun.get(x)) - conf_fun.get(x)

    # print(numerator, denominator, p)
    return numerator / denominator - p


def annealed_fun(x, p, conf_fun, nonconf_fun):
    numerator = x * conf_fun.get(1 - x) - (1 - x) * conf_fun.get(x)
    return numerator / (nonconf_fun.get(x) - x + numerator) - p


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
            stable.append(model_ode_d(x, p, conf_fun, nonconf_fun))
    return x_fixed, stable


def time_evolution_fig(fig, ax, conf_fun, nonconf_fun, p, t_max, num, xs_ini):
    t_eval = np.linspace(0, t_max, num)
    ax.plot([0, t_max], [0.5, 0.5], "k:")
    ax.set_title(conf_fun.__str__())
    for x_ini in xs_ini:
        t, x = get_time_evolution(p=p,
                                  t_max=t_max,
                                  num=num,
                                  c_ini=[x_ini],
                                  conf_fun=conf_fun,
                                  nonconf_fun=nonconf_fun,
                                  is_quenched=False)
        ax.plot(t, x, 'r')

        t, x1, x2, x = get_time_evolution(p=p,
                                          t_max=t_max,
                                          num=num,
                                          c_ini=[x_ini, x_ini],
                                          conf_fun=conf_fun,
                                          nonconf_fun=nonconf_fun,
                                          is_quenched=True)
        ax.plot(t, x, 'b--')

    ## annaled
    x_fixed, stable = get_fixed_points_for(p, conf_fun, nonconf_fun, False)
    for i, x in enumerate(x_fixed):
        if stable[i]:
            ax.plot(t_max - 1, x, "or")
        else:
            ax.plot(1, x, "sr")

    # quenched
    x_fixed, stable = get_fixed_points_for(p, conf_fun, nonconf_fun, True)
    for i, x in enumerate(x_fixed):
        if stable[i]:
            ax.plot(t_max, x, "ob")
        else:
            ax.plot(0, x, "sb")

    ax.set_xlabel("MCS")
    ax.set_ylabel("x")
    ax.set_xlim([0, t_max])
    ax.set_ylim([0, 1])


def time_evolution_que_fig(fig, ax, conf_fun, nonconf_fun, p, t_max, num, xs_ini):
    t_eval = np.linspace(0, t_max, num)
    ax.plot([0, t_max], [0.5, 0.5], "k:")
    ax.set_title(conf_fun.__str__())
    t, x1, x2, x = get_time_evolution(p=p,
                                      t_max=t_max,
                                      num=num,
                                      c_ini=[xs_ini, xs_ini],
                                      conf_fun=conf_fun,
                                      nonconf_fun=nonconf_fun,
                                      is_quenched=True)
    ax.plot(t, x, 'k')
    ax.plot(t, x1, 'r')
    ax.plot(t, x2, 'g')
    ax.set_xlabel("MCS")
    ax.set_ylabel("x")
    ax.set_xlim([0, t_max])
    ax.set_ylim([0, 1])


def quiver_fig(fig, ax, conf_fun, nonconf_fun, p):
    x, y = np.meshgrid(np.linspace(0, 1, 10), np.linspace(0, 1, 10))
    v_field = model_quenched_ode([], [x, y], p, conf_fun, nonconf_fun)
    ax.quiver(x, y, v_field[0], v_field[1])

    ax.set_title(conf_fun.__str__())
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")

    x_fixed, stable = get_fixed_points_for(p, conf_fun, nonconf_fun, True)
    for i, x in enumerate(x_fixed):
        x1 = nonconf_fun.get(x)
        x2 = conf_fun.get(x) / (conf_fun.get(x) + conf_fun.get(1 - x))
        if stable[i]:
            ax.plot(x1, x2, "og")
        else:
            ax.plot(x1, x2, "sm")

    x1_ini = np.linspace(0, 1, 14)
    for x1_i in x1_ini:
        t, x1, x2, x = get_time_evolution(p=p,
                                          t_max=200,
                                          num=400,
                                          c_ini=[x1_i, x1_i],
                                          conf_fun=conf_fun,
                                          nonconf_fun=nonconf_fun,
                                          is_quenched=True)
        ax.plot(x1, x2, 'g')

##
# q = 3
# x0, k, m = 0.5, 30, 0.5
# nonconf_fun = Logistic(x0=x0, k=k, m=m)
# ##
# p = 0.187
#
# fig, ax = plt.subplots(2, 1)
# diagram_ann_que_fig(fig=fig,
#                     ax=ax,
#                     num=200,
#                     q=q,
#                     nonconf_fun=nonconf_fun)
#
# ax[0].plot([p, p], [0, 1], ':k')
# ax[1].plot([p, p], [0, 1], ':k')
#
# conf_fun = Power(q=q)
#
# fig, ax = plt.subplots(2, 1)
#
# fig.suptitle(f"p={p} " + nonconf_fun.__str__())
# time_evolution_fig(fig,
#                    ax[0],
#                    conf_fun,
#                    nonconf_fun,
#                    p=p,
#                    t_max=100,
#                    num=200,
#                    xs_ini=np.linspace(0, 1, 10))
#
# conf_fun = SymmetricPower(q=q)
# time_evolution_fig(fig,
#                    ax[1],
#                    conf_fun,
#                    nonconf_fun,
#                    p=p,
#                    t_max=100,
#                    num=200,
#                    xs_ini=np.linspace(0, 1, 10))
# plt.tight_layout()
#
# fig, ax = plt.subplots(2, 1)
# fig.suptitle(f"p={p} " + nonconf_fun.__str__())
#
# conf_fun = Power(q=q)
# quiver_fig(fig, ax[0], conf_fun, nonconf_fun, p)
#
# conf_fun = SymmetricPower(q=q)
# quiver_fig(fig, ax[1], conf_fun, nonconf_fun, p)
# plt.tight_layout()
#
# fig, ax = plt.subplots(2, 1)
# fig.suptitle(f"p={p} " + nonconf_fun.__str__())
#
# conf_fun = Power(q=q)
# time_evolution_que_fig(fig, ax[0], conf_fun, nonconf_fun, p, 100, 100, 1)
# conf_fun = SymmetricPower(q=q)
# time_evolution_que_fig(fig, ax[1], conf_fun, nonconf_fun, p, 100, 100, 1)
# plt.tight_layout()
#
# plt.show()
