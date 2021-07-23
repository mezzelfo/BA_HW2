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

# problem = Problem.from_multiple(
#     demand_probs=[
#         [1,2,3,4,5,6],
#         [6,5,4,3,2,1],
#     ],
#     max_inv=40,
#     make=12
# )



V, Q = problem.value_iteration(gamma=0.99)
mu = Q.argmin(0)

np.save('mu_opt.npy',mu.cpu().numpy())
exit()

# import matplotlib.pyplot as plt
# fig = plt.figure()
# ax = fig.add_subplot(111, projection = '3d')

# plt.matshow(mu[0,:,:,2],vmin=0,vmax=3)
# plt.colorbar()
# plt.matshow(mu[0,:,:,18],vmin=0,vmax=3)
# plt.colorbar()

# inv = torch.arange(0,20+1)
# X, Y, Z = torch.meshgrid(inv,inv,inv)
# list_of_pts = torch.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
# mulist = mu[0].ravel()
# list_of_pts = list_of_pts[mulist > 0]
# mulist = mulist[mulist > 0]
# print(list_of_pts.shape)
# ax.contour(X,Y,Z, c = mu)
# plt.show()

# np.savetxt('V.csv',V.cpu().numpy().ravel(),delimiter = ',')
# np.savetxt('Q.csv',Q.cpu().numpy().ravel(),delimiter = ',')
# np.savetxt('mu.csv',mu.cpu().numpy().ravel(),delimiter = ',')

# np.save('mu.npy',mu.cpu().numpy())
# np.save('Q.npy',Q.cpu().numpy())
# np.save('V.npy',V.cpu().numpy())

exit()

init_inventory = [1,1,1,1]
init_machine = 0
vtrue = V[tuple([init_machine])+tuple(init_inventory)].item()
print(vtrue)

res = problem.simulate(
    init_inventory, init_machine, mu.cpu(),
    time_steps = 500, repetitions = 1000, gamma = 0.99
)
print(res.mean(), res.min(), res.max(), res.shape)


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