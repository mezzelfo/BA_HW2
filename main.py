from withstatus import Problem
import numpy as np

problem = Problem.from_multiple(
    demand_probs=[
        [6,5,4,3,2,1],
        [1,2,3,4,5,6],
    ],
    max_inv=100,
    make=14
)

V, Q = problem.value_iteration(gamma=0.99)
mu = Q.argmin(0)

np.savetxt('V.csv',V.cpu().numpy().ravel(),delimiter = ',')
np.savetxt('Q.csv',Q.cpu().numpy().ravel(),delimiter = ',')
np.savetxt('mu.csv',mu.cpu().numpy().ravel(),delimiter = ',')

init_inventory = [5,5]
init_machine = 1
vtrue = V[tuple([init_machine])+tuple(init_inventory)].item()
print(vtrue)

res = problem.simulate(
    init_inventory, init_machine, mu.cpu()
)
print(res.mean(), res.min(), res.max(), res.shape)
import matplotlib.pyplot as plt

plt.hist(res,density=True,bins=50)
ymin, ymax = plt.ylim()
plt.xlim(V.min().item(),V.max().item())
plt.vlines(vtrue,ymin, ymax,colors='red',label='true')
plt.vlines(res.mean(),ymin, ymax,colors='green',linestyles='dashed',label='sample mean')
plt.legend()
plt.show()