import matplotlib.pyplot as plt
from theoretical_module import get_fixed_points, get_fixed_points_q_voter

num = 100
q = 3
f = 0.5

ps, cs = get_fixed_points(num=num,
                          q=q,
                          f=f,
                          is_quenched=False)
plt.figure(1)
plt.plot(ps, cs, '-')
ps, cs = get_fixed_points(num=num,
                          q=q,
                          f=f,
                          is_quenched=True)
plt.plot(ps, cs, '-')
plt.xlim([0, 1])
plt.show()
