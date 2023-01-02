import matplotlib.pyplot as plt
from theoretical_module import get_fixed_points, Power, Logistic

q = 1.5
x0 = 0.5
k = 0
m = 0.6
is_quenched = True

conf_fun = Power(q=q)
print(conf_fun)

nonconf_fun = Logistic(x0=x0, k=k, m=m)
print(nonconf_fun)

ps, cs = get_fixed_points(100, conf_fun, nonconf_fun, is_quenched=is_quenched)
plt.plot(ps, cs)
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.title(conf_fun.__str__() + " " + nonconf_fun.__str__() + " " + f"is_quenched={is_quenched}")
plt.xlabel("nonconformity")
plt.ylabel("concentration")
plt.show()
