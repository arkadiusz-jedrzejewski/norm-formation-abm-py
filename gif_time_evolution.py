import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
from theoretical_module import Logistic, Power, SymmetricPower, get_fixed_points, get_roots
from time_evolution import diagram_ann_que_fig, time_evolution_fig, quiver_fig, diagram_fig

##
q = 7
x0, k, m = 0.5, 15, 0.5
p = 0.05

nonconf_fun = Logistic(x0=x0, k=k, m=m)
conf_fun = SymmetricPower(q=q)

title = f"Time-evolution " + conf_fun.__str__() + " " + nonconf_fun.__str__()
metadata = dict(title=title, artist="Arek")
writer = PillowWriter(fps=5, metadata=metadata)
##
fig, ax = plt.subplots(1, 3)
fig.set_size_inches(15, 7)
pp, _ = diagram_fig(ax[0],
                            col="r",
                            num=200,
                            conf_fun=conf_fun,
                            nonconf_fun=nonconf_fun,
                            is_quenched=False)
p_max = max(pp)
pp, _ = diagram_fig(ax[0],
                            col="b",
                            num=200,
                            conf_fun=conf_fun,
                            nonconf_fun=nonconf_fun,
                            is_quenched=True)
p_max = max(p_max, max(pp))

with writer.saving(fig, title + ".gif", 100):
    for p in np.linspace(0.01, p_max*1.05, 200):
        pp, _ = diagram_fig(ax[0],
                            col="r",
                            num=200,
                            conf_fun=conf_fun,
                            nonconf_fun=nonconf_fun,
                            is_quenched=False)
        p_max = max(pp)
        pp, _ = diagram_fig(ax[0],
                            col="b",
                            num=200,
                            conf_fun=conf_fun,
                            nonconf_fun=nonconf_fun,
                            is_quenched=True)
        p_max = max(p_max, max(pp))
        ax[0].set_ylim([0, 1])
        ax[0].set_ylabel("$x^*$")
        ax[0].set_xlabel("$p$")
        ax[0].set_title(conf_fun.__str__())

        ax[0].set_xlim([0, min(p_max * 1.05, 1)])
        ax[0].plot([p, p], [0, 1], "k:")
        ax[0].plot([0, 1], [0.5, 0.5], "k:")
        time_evolution_fig(fig,
                           ax[1],
                           conf_fun,
                           nonconf_fun,
                           p=p,
                           t_max=50,
                           num=200,
                           xs_ini=np.linspace(0, 1, 20))
        quiver_fig(fig, ax[2], conf_fun, nonconf_fun, p)

        # conf_fun = SymmetricPower(q=q)
        # p_max_2 = diagram_ann_que_fig(fig=fig,
        #                               ax=ax[1, 0],
        #                               num=200,
        #                               conf_fun=conf_fun,
        #                               nonconf_fun=nonconf_fun)
        # ax[1, 0].plot([p, p], [0, 1], "k:")
        # time_evolution_fig(fig,
        #                    ax[1, 1],
        #                    conf_fun,
        #                    nonconf_fun,
        #                    p=p,
        #                    t_max=100,
        #                    num=200,
        #                    xs_ini=np.linspace(0, 1, 10))
        # quiver_fig(fig, ax[1, 2], conf_fun, nonconf_fun, p)

        # ax[0, 0].set_xlim([0, max(p_max_1, p_max_2) * 1.03])
        # ax[1, 0].set_xlim([0, max(p_max_1, p_max_2) * 1.03])
        fig.suptitle(f"p={p:.4f} " + nonconf_fun.__str__())
        plt.tight_layout()
        writer.grab_frame()
        print(p)
        ax[0].cla()
        ax[1].cla()
        ax[2].cla()
