from withstatus import Problem
import numpy as np
import torch
import matplotlib.pyplot as plt

problem = Problem.from_multiple(
    demand_probs=[
        [1,2,3,4,5,6],
        [6,5,4,3,2,1],
        [1,2,3,4,5,0],
        [5,4,3,2,1,0],
    ],
    max_inv=20,
    make=4*4
)

#V, Q = problem.value_iteration(gamma=0.99)
V2, Q2 = problem.generic_iteration_directly_decomposed_pytorch(gamma=0.99)

#print((Q-Q2).abs().max())
exit()

#mu = Q.argmin(0)

#np.save('mu_opt.npy',mu.cpu().numpy())
#exit()

# init_time_bucket = 0
# init_machine = 0
# init_inventory = [20,20,20,20]
# vtrue = V[tuple([init_time_bucket,init_machine])+tuple(init_inventory)].item()
# print(vtrue)

# res = problem.simulate(
#     init_inventory, init_machine, init_time_bucket, mu.cpu(),
#     time_steps = 1000, repetitions = 1000, gamma = 0.99
# )
# print(res.mean(), res.min(), res.max(), res.shape)


# mu = mu.cpu().numpy()
# plt.subplot(1,3,1)
# plt.matshow(mu[0],vmin=0,vmax=2,fignum=False)
# plt.subplot(1,3,2)
# plt.matshow(mu[1],vmin=0,vmax=2,fignum=False)
# plt.subplot(1,3,3)
# plt.matshow(mu[2],vmin=0,vmax=2,fignum=False)
# plt.show()

plt.hist(res,density=True,bins=50)
ymin, ymax = plt.ylim()
#plt.xlim(V.min().item(),V.max().item())
plt.vlines(vtrue,ymin, ymax,colors='red',label='true')
plt.vlines(res.mean(),ymin, ymax,colors='green',linestyles='dashed',label='sample mean')
plt.legend()
plt.show()