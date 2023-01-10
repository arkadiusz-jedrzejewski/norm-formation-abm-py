import matplotlib.pyplot as plt
import numpy as np

from theoretical_module import get_fixed_points, Power, SymmetricPower, Logistic

q = 2
x0 = 0.5
k = 14.5
m = 0.5

conf_fun = Power(q=q)
print(conf_fun)

nonconf_fun = Logistic(x0=x0, k=k, m=m)
print(nonconf_fun)

ps, cs = get_fixed_points(100, conf_fun, nonconf_fun, is_quenched=True)
plt.plot(1 - ps, cs)

ps, cs = get_fixed_points(100, conf_fun, nonconf_fun, is_quenched=False)
plt.plot(1 - ps, cs, '--r')

plt.legend(["quenched", "annealed"])
plt.title(conf_fun.__str__() + " " + nonconf_fun.__str__())
plt.xlabel("nonconformity")
plt.ylabel("concentration")
plt.xlim([0, 1])
plt.ylim([0, 1])

plt.plot(1 - 4 * (q - 1) / (4 * q + k), 0.5,
         '.k')  # for SymmetricPower (=> annealed = quenched) / for Power quenched case
plt.plot(1 - (q - 1) / (q - 1 + 2 ** (q - 1) * (k / 4 + 1)), 0.5, '.r')  # for Power annealed case
plt.show()

plt.figure()
# for Power annealed case
q = np.linspace(1.01, 10, 1000)
k = ((40 * q) / 3 - (8 * q ** 2) / 3) / (
        80 * q + ((80 * q - 16 * q ** 2) ** 2 - ((40 * q) / 3 - (8 * q ** 2) / 3) ** 3) ** (
        1 / 2) - 16 * q ** 2) ** (1 / 3) + (
            80 * q + ((- 16 * q ** 2 + 80 * q) ** 2 - ((40 * q) / 3 - (8 * q ** 2) / 3) ** 3) ** (
            1 / 2) - 16 * q ** 2) ** (1 / 3)
plt.plot(q, k)
plt.show()

plt.figure()
# for Power quenched case
k = np.linspace(-1, 30, 1000)
q = -(2 * k - ((k + 2) * (k + 4) * (k ** 2 - 2 * k + 8)) ** (1 / 2) + 8) / (4 * (k + 4))
plt.plot(q, k)
q = np.linspace(0, 8, 1001)
k = ((16 * q ** 2) / 3 + (16 * q) / 3) / (
            32 * q + ((32 * q ** 2 + 32 * q) ** 2 - ((16 * q ** 2) / 3 + (16 * q) / 3) ** 3) ** (1 / 2) + 32 * q ** 2) ** (
                1 / 3) + (
                32 * q + ((32 * q ** 2 + 32 * q) ** 2 - ((16 * q ** 2) / 3 + (16 * q) / 3) ** 3) ** (1 / 2) + 32 * q ** 2) ** (
                1 / 3)
plt.plot(q, k, '--r')
k = (q + 1) * 4
plt.plot(q, k, '--g')
plt.show()
