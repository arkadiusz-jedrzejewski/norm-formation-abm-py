import matplotlib.pyplot as plt

from theoretical_module import *
import tikzplotlib
from matplotlib.animation import PillowWriter

# plot_diagram_symmetric_power_fun_k(q=4, ks=np.linspace(0, 40, 100))
# plot_diagram_symmetric_power_fun_q(qs=np.linspace(1.0001, 8, 3000), k=30)
# plot_diagram_symmetric_power(qs=np.linspace(1.0001, 8, 100), ks=np.linspace(0, 40, 100))
# plot_diagram_power_fun_k(q=1.2, ks=np.linspace(-3, 40, 100), is_quenched=False)
# plot_diagram_power_fun_q(qs=np.linspace(1.0001, 8, 100), k=8, is_quenched=False)
# plot_diagram_power_fun_k(q=5, ks=np.linspace(-3, 40, 100), is_quenched=True)
# plot_diagram_power_fun_q(qs=np.linspace(1.0001, 8, 100), k=30, is_quenched=True)
# plot_diagram_power(q=np.linspace(0, 10, 1000), k=np.linspace(-4, 36, 1000), is_quenched=True)
# plot_diagram_power(q=np.linspace(0, 10, 1000), k=np.linspace(-4, 36, 1000), is_quenched=False)


# ## movie plot
#
#
# # Initialize the movie
fig = plt.figure()
line, = plt.plot([0, 1], [0.5, 0.5], ':k')
que, = plt.plot([], [], label="quenched")
ann, = plt.plot([], [], '--r', label="annealed")
plt.legend()
plt.xlim([0, 1])
plt.ylim([0, 1])

q = 8
k = -4
x0 = 0.5
m = 0.5

conf_fun = Power(q=q)
nonconf_fun = Logistic(x0=x0, k=k, m=m)
ps, cs = get_fixed_points(1000, conf_fun, nonconf_fun, is_quenched=True)
que.set_data(ps, cs)

ps, cs = get_fixed_points(1000, conf_fun, nonconf_fun, is_quenched=False)
ann.set_data(ps, cs)
plt.xlabel("p")
plt.ylabel("$x^*$")
plt.title(conf_fun.__str__() + " " + nonconf_fun.__str__())
plt.show()
#
# title = f"SymmetricPower-{q}q-both"
# metadata = dict(title=title, artist="Arek")
# writer = PillowWriter(fps=15, metadata=metadata)
#
# with writer.saving(fig, title + ".gif", 100):
#     for k in np.geomspace(1, 55, 500)-5:
#         nonconf_fun = Logistic(x0=x0, k=k, m=m)
#         ps, cs = get_fixed_points(1000, conf_fun, nonconf_fun, is_quenched=True)
#         que.set_data(ps, cs)
#
#         ps, cs = get_fixed_points(1000, conf_fun, nonconf_fun, is_quenched=False)
#         ann.set_data(ps, cs)
#         plt.title(f"{conf_fun.__str__()}  {nonconf_fun.__str__()}")
#         writer.grab_frame()


## movie plot

# #
# # # Initialize the movie
# fig, axs = plt.subplots(2, figsize=(7, 10))
# line, = axs[0].plot([0, 1], [0.5, 0.5], ':k')
# que, = axs[0].plot([], [], label="quenched")
# ann, = axs[0].plot([], [], 'r', label="annealed")
# axs[0].legend(loc='upper right')
# axs[0].set_xlim([0, 1])
# axs[0].set_ylim([0, 1])
# axs[0].set_xlabel("p")
# axs[0].set_ylabel("$x^*$")
#
# line, = axs[1].plot([0, 1], [0.5, 0.5], ':k')
# que2, = axs[1].plot([], [], label="quenched")
# ann2, = axs[1].plot([], [], '--r', label="annealed")
# axs[1].legend(loc='upper right')
# axs[1].set_xlim([0, 1])
# axs[1].set_ylim([0, 1])
# axs[1].set_xlabel("p")
# axs[1].set_ylabel("$x^*$")
#
# q = 8
# k = -4
# x0 = 0.5
# m = 0.5
#
# title = f"Diagrams-{q}q-both"
# metadata = dict(title=title, artist="Arek")
# writer = PillowWriter(fps=15, metadata=metadata)
#
# with writer.saving(fig, title + ".gif", 100):
#     for k in np.geomspace(1, 55, 500) - 5:
#         conf_fun = Power(q=q)
#         nonconf_fun = Logistic(x0=x0, k=k, m=m)
#
#         fig.suptitle(nonconf_fun.__str__())
#
#         ps, cs = get_fixed_points(1000, conf_fun, nonconf_fun, is_quenched=True)
#         que.set_data(ps, cs)
#
#         ps, cs = get_fixed_points(1000, conf_fun, nonconf_fun, is_quenched=False)
#         ann.set_data(ps, cs)
#         axs[0].set_title(f"{conf_fun.__str__()}")
#
#         conf_fun = SymmetricPower(q=q)
#         ps, cs = get_fixed_points(1000, conf_fun, nonconf_fun, is_quenched=True)
#         que2.set_data(ps, cs)
#
#         ps, cs = get_fixed_points(1000, conf_fun, nonconf_fun, is_quenched=False)
#         ann2.set_data(ps, cs)
#         axs[1].set_title(f"{conf_fun.__str__()}")
#         plt.tight_layout()
#         writer.grab_frame()


## movie plot 3D





# Initialize the movie
#
# q = 8
# k = -4
# x0 = 0.5
# m = 0.5
#
# fig, axs = plt.subplots(1, figsize=(7, 10), subplot_kw={'projection': '3d'})
# # # plot_diagram_power_fun_k(axs, q=q, ks=np.linspace(-4, 50, 100), is_quenched=True)
#
# # que, = axs.plot3D([], [], [], label="quenched")
# # axs.set_xlim([-4, 50])
# # axs.set_ylim([0, 1])
# # axs.set_zlim([0, 1])
# # axs.set_xlabel("k")
# # axs.set_ylabel("p")
# # axs.set_zlabel("$x^*$")
#
# # plot_diagram_power_fun_k(axs, q=q, ks=np.linspace(-4, 50, 100), is_quenched=False)
# #
# # ann, = axs.plot3D([], [], [], label="annealed")
# # axs.set_xlim([-4, 50])
# # axs.set_ylim([0, 1])
# # axs.set_zlim([0, 1])
# # axs.set_xlabel("k")
# # axs.set_ylabel("p")
# # axs.set_zlabel("$x^*$")
# #
# plot_diagram_symmetric_power_fun_k(axs, q=q, ks=np.linspace(-4, 50, 100))
# symm, = axs.plot3D([], [], [], label="annealed")
# axs.set_xlim([-4, 50])
# axs.set_ylim([0, 1])
# axs.set_zlim([0, 1])
# axs.set_xlabel("k")
# axs.set_ylabel("p")
# axs.set_zlabel("$x^*$")
#
# title = f"Diagrams3D-{q}q-both"
# metadata = dict(title=title, artist="Arek")
# writer = PillowWriter(fps=15, metadata=metadata)
#
# with writer.saving(fig, title + ".gif", 100):
#     for k in np.geomspace(1, 55, 50) - 5:
#         conf_fun = Power(q=q)
#         nonconf_fun = Logistic(x0=x0, k=k, m=m)
#
#         fig.suptitle(nonconf_fun.__str__())
#
#         # ps, cs = get_fixed_points(1000, conf_fun, nonconf_fun, is_quenched=True)
#         # que.set_data([k] * len(ps), ps)
#         # que.set_3d_properties(cs)
#
#         # ps, cs = get_fixed_points(1000, conf_fun, nonconf_fun, is_quenched=False)
#         # ann.set_data([k] * len(ps), ps)
#         # ann.set_3d_properties(cs)
#         #
#         conf_fun = SymmetricPower(q=q)
#         ps, cs = get_fixed_points(1000, conf_fun, nonconf_fun, is_quenched=True)
#         symm.set_data([k] * len(ps), ps)
#         symm.set_3d_properties(cs)
#         #
#         # ps, cs = get_fixed_points(1000, conf_fun, nonconf_fun, is_quenched=False)
#         # ann2.set_data(ps, cs)
#         # axs[1].set_title(f"{conf_fun.__str__()}")
#         plt.tight_layout()
#         writer.grab_frame()
#
# # with writer.saving(fig, "writer_test.mp4", 100):
# #     for i in range(n):
# #         x0 = x[i]
# #         y0 = y[i]
# #         red_circle.set_data(x0, y0)
# #         writer.grab_frame()
# # qd = q = 6
# # x0 = 0.5
# # kd = k = 16
# # m = 0.5
# #
# # conf_fun = Power(q=q)
# # print(conf_fun)
# #
# # plot_diagram_power_fun_k(q=q, ks=np.linspace(-3.5, 40, 100), is_quenched=False)
# # tikzplotlib.save("test.tikz")
# # nonconf_fun = Logistic(x0=x0, k=-3.5, m=m)
# # print(nonconf_fun)
# #
# # ps, cs = get_fixed_points(1000, conf_fun, nonconf_fun, is_quenched=False)
# # plt.plot([-3.5]*len(ps), ps, cs)
# #
# # nonconf_fun = Logistic(x0=x0, k=40, m=m)
# # print(nonconf_fun)
# # ps, cs = get_fixed_points(1000, conf_fun, nonconf_fun, is_quenched=False)
# # plt.plot([40]*len(ps), ps, cs, '--r')
# #
# # plt.legend(["quenched", "annealed"])
# # plt.title(conf_fun.__str__() + " " + nonconf_fun.__str__())
# # plt.xlabel("p")
# # plt.ylabel("$x^*$")
# plt.show()

# plot_diagram_symmetric_power_fun_k(q=q, ks=np.linspace(20, 40, 100))
#
#
# nonconf_fun = Logistic(x0=x0, k=20, m=m)
# print(nonconf_fun)
#
# ps, cs = get_fixed_points(200, conf_fun, nonconf_fun, is_quenched=True)
# plt.plot([20]*len(ps),ps, cs)
#
# nonconf_fun = Logistic(x0=x0, k=40, m=m)
# print(nonconf_fun)
# ps, cs = get_fixed_points(200, conf_fun, nonconf_fun, is_quenched=True)
# plt.plot([40]*len(ps), ps, cs, 'r')
#
# k=32
# nonconf_fun = Logistic(x0=x0, k=k, m=m)
# print(nonconf_fun)
# ps, cs = get_fixed_points(200, conf_fun, nonconf_fun, is_quenched=True)
# plt.plot(ps-0.3, cs*2-1, 'r')
# tikzplotlib.save("test.tikz")
# plt.show()
#
# plot_diagram_symmetric_power(qs=np.linspace(1.0001, 9, 600), ks=np.linspace(0, 50, 600))
# plt.show()
# plt.xlim([0, 1])
# plt.ylim([0, 1])

# plt.plot(4 * (q - 1) / (4 * q + k), 0.5,
#          '.c')  # for SymmetricPower (=> annealed = quenched) / for Power quenched case
# plt.plot((q - 1) / (q - 1 + 2 ** (q - 1) * (k / 4 + 1)), 0.5, '.r')  # for Power annealed case


# q = k / 4
# plt.plot(q, q * 0 - 4, 'c:')
# q = - k / 4
# plt.plot(q, k, 'k:')

# q = np.linspace(0, 8, 1001)
# k = ((16 * q ** 2) / 3 + (16 * q) / 3) / (
#         32 * q + ((32 * q ** 2 + 32 * q) ** 2 - ((16 * q ** 2) / 3 + (16 * q) / 3) ** 3) ** (1 / 2) + 32 * q ** 2) ** (
#             1 / 3) + (
#             32 * q + ((32 * q ** 2 + 32 * q) ** 2 - ((16 * q ** 2) / 3 + (16 * q) / 3) ** 3) ** (
#             1 / 2) + 32 * q ** 2) ** (
#             1 / 3)
# plt.plot(q, k, '--r')
# k = (q + 1) * 4
# plt.plot(q, k, '--g')
# plt.plot([qd, qd], [min(k), max(k)], ":m")
# plt.plot([min(q), max(q)], [kd, kd], ":m")
# plt.plot(qd, kd, '.m')
