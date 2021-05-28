import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.linalg

production_makes = torch.tensor([[0,0],[12,0],[0,12]])
production_setup = torch.tensor([[0,0],[6,0],[0,6]])
setup_cost = 40.0
demandsDistributions = torch.distributions.Categorical(torch.tensor([[1,2,3,4,5,6],[6,5,4,3,2,1]]))
max_inv = 50
L = 100000
gamma = 0.99

def sample_transitions(initial_machine_state, initial_inventory_state, action):
    # initial_machine_state:    tensor of size L
    # initial_inventory_state:  tensor of size LxN
    # action:                   tensor of size L
    demds = demandsDistributions.sample((L,)) #tensor of size LxN
    is_coherent = (action == initial_machine_state)
    orders = torch.where(is_coherent.unsqueeze(1), production_makes[action], production_setup[action])
    inv_state = initial_inventory_state+orders
    cost = torch.where((is_coherent.logical_not()) & (action > 0), setup_cost, 0.0)
    exceeded = torch.where(demds > inv_state, demds-inv_state, 0)
    cost += exceeded.sum(1)*100
    inv_state -= demds
    inv_state = torch.clamp(inv_state, 0, max_inv)
    cost += inv_state.sum(1)
    return torch.clone(action), inv_state, cost

def evaluate_basis_functions(machine_state, inv_state, action):
    # machine_state:    tensor of size L
    # inv_state:        tensor of size LxN
    # action:           tensor of size L
    columns = []
    for a in range(3):
        for m in range(3):
            for v in [
                1,
                inv_state[:,0], inv_state[:,1],
                (inv_state[:,0]-inv_state[:,1]).abs(),
                inv_state[:,0]/(inv_state[:,1]+1),inv_state[:,1]/(inv_state[:,0]+1)
                #inv_state[:,0]**2, inv_state[:,1]**2, inv_state[:,0]*inv_state[:,1],
                #torch.exp(-torch.maximum(inv_state[:,0],inv_state[:,1])/10)
                #inv_state[:,0]**3, inv_state[:,1]**3, inv_state[:,0]**2*inv_state[:,1],inv_state[:,0]*inv_state[:,1]**2,
                #torch.exp(-(inv_state[:,0]**2+10*inv_state[:,1])/50),
                #torch.exp(-(10*inv_state[:,0]+inv_state[:,1]**2)/50),
                #1/(inv_state[:,0]+20),1/(inv_state[:,1]+20),1/(inv_state[:,0]+inv_state[:,1]+30)
                ]:
                c = 1.0 * v * (action == a) * (machine_state == m)
                columns.append(
                    c
                )
    
    M = torch.vstack(columns).T
    return M

def evalute_policy(w, machine_state, inv_state):
    m0 = evaluate_basis_functions(machine_state, inv_state, 0*torch.ones_like(machine_state))
    m1 = evaluate_basis_functions(machine_state, inv_state, torch.ones_like(machine_state))
    m2 = evaluate_basis_functions(machine_state, inv_state, 2*torch.ones_like(machine_state))
    q0 = m0 @ w #(L x m) @ m
    q1 = m1 @ w
    q2 = m2 @ w
    Q = torch.vstack([q0,q1,q2]) # (N+1) x L
    return Q.argmin(0) # L

def get_policy(machine_state, w):
    inv = torch.arange(0,max_inv+1)
    X, Y = torch.meshgrid(inv,inv)
    inv_state = torch.vstack([X.ravel(), Y.ravel()]).T
    mu = evalute_policy(w, torch.ones(((max_inv+1)*(max_inv+1),))*machine_state, inv_state)
    mu = mu.reshape((max_inv+1,max_inv+1))
    return mu

def validate_policy(policy):
    machine_state = torch.randint(0,2+1,(L,))
    inventory_state = torch.randint(0,max_inv+1,(L,2))

    running_cost = torch.zeros_like(machine_state, dtype=torch.float)
    for t in range(500):
        actions = policy[tuple([machine_state.T])+tuple(inventory_state.T)]
        machine_state, inventory_state, cost = sample_transitions(machine_state, inventory_state, actions)
        running_cost = gamma*running_cost + cost

    # print(running_cost.mean(),running_cost.std())
    # plt.hist(running_cost.cpu().numpy(),bins=100)
    return running_cost.mean()
    
# def get_action(machine_state, inventory_state, coeffs):
#     # machine_state:    tensor of size L
#     # inv_state:        tensor of size LxN
#     # coeffs:           tensor of size (N+1) x N x 2
#     inv_state = inventory_state/max_inv
#     actions = torch.zeros_like(machine_state, dtype=torch.long)
#     for l in range(len(machine_state)):
#         c = coeffs[machine_state[l],:,:]
#         if inv_state[l,0] > c[0,0] and inv_state[l,1] > c[1,0]:
#             actions[l] = 0

#         elif inv_state[l,0] > c[0,0] and inv_state[l,1] > c[1,1]:
#             actions[l] = 0
#         elif inv_state[l,0] > c[0,1] and inv_state[l,1] > c[1,0]:
#             actions[l] = 0

#         elif inv_state[l,0] > c[0,0] and inv_state[l,1] < c[1,1]:
#             actions[l] = 2
#         elif inv_state[l,0] < c[0,1] and inv_state[l,1] > c[1,0]:
#             actions[l] = 1
#         else:
#             actions[l] = (inv_state[l,:]/c[:,0]).argmin()+1
#     return actions

def get_action_pytorch(machine_state, inventory_state, coeffs):
    # coeffs:   tensor of size (N+1) x N with scalars in [0,1]
    inv_state = inventory_state/max_inv
    shortage = coeffs[machine_state,:]
    actions = torch.where(
        torch.all(inv_state > shortage, 1), 0,
        (inv_state / shortage).argmin(1)+1
        )
    return actions


def get_loss(coeffs):
    # numpy array flattened
    c = torch.from_numpy(coeffs).reshape(2+1,2)
    # coeffs:   tensor of size (N+1) x N with scalars in [0,1]
    machine_state = torch.randint(0,2+1,(L,))
    inventory_state = torch.randint(0,max_inv+1,(L,2))
    running_cost = torch.zeros_like(machine_state, dtype=torch.float)
    for t in range(500):
        actions = get_action_pytorch(machine_state, inventory_state, c)
        machine_state, inventory_state, cost = sample_transitions(machine_state, inventory_state, actions)
        running_cost = gamma*running_cost + cost
    #print(coeffs,running_cost.mean().item())
    return running_cost.mean()

def tabulate_mu_from_coeffs(res):
    inv = torch.arange(0,max_inv+1)
    X, Y = torch.meshgrid(inv,inv)
    inv_state = torch.vstack([X.ravel(), Y.ravel()]).T
    mu0 = get_action_pytorch(torch.ones(((max_inv+1)*(max_inv+1),),dtype=torch.long)*0, inv_state, torch.from_numpy(res.reshape(2+1,2)))
    mu0 = mu0.reshape((max_inv+1,max_inv+1))
    mu1 = get_action_pytorch(torch.ones(((max_inv+1)*(max_inv+1),),dtype=torch.long)*1, inv_state, torch.from_numpy(res.reshape(2+1,2)))
    mu1 = mu1.reshape((max_inv+1,max_inv+1))
    mu2 = get_action_pytorch(torch.ones(((max_inv+1)*(max_inv+1),),dtype=torch.long)*2, inv_state, torch.from_numpy(res.reshape(2+1,2)))
    mu2 = mu2.reshape((max_inv+1,max_inv+1))
    return torch.stack([mu0,mu1,mu2])



# import numpy as np

# from geneticalgorithm import geneticalgorithm as ga

# algorithm_param = {'max_num_iteration': 10,\
#                    'population_size':100,\
#                    'mutation_probability':0.1,\
#                    'elit_ratio': 0.1,\
#                    'crossover_probability': 0.5,\
#                    'parents_portion': 0.3,\
#                    'crossover_type':'uniform',\
#                    'max_iteration_without_improv':None}

# model=ga(function=get_loss,dimension=(2+1)*2,variable_type='real',variable_boundaries=np.array([[0,1]]*((2+1)*2)),algorithm_parameters=algorithm_param)
# model.run()
# res = model.output_dict['variable']
# mu = tabulate_mu_from_coeffs(res)
# muopt = torch.LongTensor(np.load('mu.npy'))

# print('optimal mu value:',validate_policy(muopt))
# print('heuristic mu value:',validate_policy(mu))


# plt.subplot(2,3,1)
# plt.matshow(muopt[0],vmin=0,vmax=2,fignum=False)
# plt.subplot(2,3,2)
# plt.matshow(muopt[1],vmin=0,vmax=2,fignum=False)
# plt.subplot(2,3,3)
# plt.matshow(muopt[2],vmin=0,vmax=2,fignum=False)
# plt.subplot(2,3,4)
# plt.matshow(mu[0],vmin=0,vmax=2,fignum=False)
# plt.subplot(2,3,5)
# plt.matshow(mu[1],vmin=0,vmax=2,fignum=False)
# plt.subplot(2,3,6)
# plt.matshow(mu[2],vmin=0,vmax=2,fignum=False)
# plt.show()



#exit()

# w = 1+torch.rand((54,))*100
# muopt = torch.LongTensor(np.load('mu.npy'))
# for k in range(20):
#     print(k)
#     machine_state = torch.randint(0,2+1,(L,))
#     inventory_state = torch.randint(0,max_inv+1,(L,2))
#     random_action = torch.randint(0,2+1,(L,))

#     A = evaluate_basis_functions(machine_state, inventory_state, random_action)

#     b = torch.zeros_like(machine_state, dtype=torch.float)
#     for t in range(200):
#         #print('\t',t)
#         if t == 0:
#             actions = random_action
#         else:
#             actions = evalute_policy(w, machine_state, inventory_state)
#         machine_state, inventory_state, cost = sample_transitions(machine_state, inventory_state, actions)
#         b += cost * gamma**t

#     #w = 0.8*w+ 0.2*(torch.linalg.pinv(A) @ b)
#     #w = torch.linalg.pinv(A) @ b
#     wnew = torch.lstsq(b.unsqueeze(1),A)[0][:54,:].squeeze()
#     w = 0.9*w+ 0.1*wnew
    
#     #w = w.squeeze()
# print(w)
# plt.figure()
# validate_policy(muopt)
# #mu = torch.stack([get_policy(0,w),get_policy(1,w),get_policy(2,w)])
# mu = torch.randint_like(muopt,high=2+1)
# plt.figure()
# validate_policy(mu)
# plt.show()
# exit()



# Qopt = torch.tensor(np.load('Q.npy'))
# muopt = torch.LongTensor(np.load('mu.npy'))
# machine_state = torch.randint(0,2+1,(L,))
# inventory_state = torch.randint(0,max_inv+1,(L,2))
# random_action = torch.randint(0,2+1,(L,))
# print(muopt.shape, Qopt.shape)
# b = Qopt[tuple([random_action.T])+tuple([machine_state.T])+tuple(inventory_state.T)]
# A = evaluate_basis_functions(machine_state, inventory_state, random_action)
# w = torch.linalg.pinv(A) @ b
# print(A.shape, b.shape, w.shape)
# res = A@w-b
# print(res.abs().max(), b.abs().max())
# print(w.reshape(3,3,-1).max(0)[0].max(0)[0].int())

w = torch.tensor([2915.828458529534069, 1.476027081457305641, 5.40926076518791032, \
-0.1359797797755389624, 8.79529108080419708, 11.75247627516831671, \
2915.828458529534069, 1.476027081457305641, 5.40926076518791032, \
-0.1359797797755389624, 8.79529108080419708, 11.75247627516831671, \
2915.828458529534069, 1.476027081457305641, 5.40926076518791032, \
-0.1359797797755389624, 8.79529108080419708, 11.75247627516831671, \
2882.170993150436360, 3.35214268918578535, 6.82922762550312054, \
0.706639135267252570, 9.34313347077361689, 0.1285525144469746086, \
2847.621138642220661, 4.03016322990036088, 6.97329203642037231, \
0.448693416789403120, 9.75953352844869986, 0.354497766400438208, \
2882.170993150436360, 3.35214268918578535, 6.82922762550312054, \
0.706639135267252570, 9.34313347077361689, 0.1285525144469746086, \
2900.385951580898011, 2.379422800116360001, 8.12456925146579516, \
0.2943095197410446072, 3.15787685594806840, 12.36525564221722193, \
2900.385951580898011, 2.379422800116360001, 8.12456925146579516, \
0.2943095197410446072, 3.15787685594806840, 12.36525564221722193, \
2887.113032528545502, 2.638257387123304122, 8.99420470773461196, \
-0.333673896877783565, 2.502117211947418044, 13.33731819954744411])
muopt = torch.LongTensor(np.load('mu.npy'))
for k in range(20):
    print(k)
    machine_state = torch.randint(0,2+1,(L,))
    inventory_state = torch.randint(0,max_inv+1,(L,2))
    random_action = torch.randint(0,2+1,(L,))
    #quasi_policy_action = evalute_policy(w,initial_machine_state, initial_inventory_state)
    #eps = torch.rand(quasi_policy_action.shape)
    #quasi_policy_action = torch.where(eps < 100, torch.randint(0,2+1,(L,)), quasi_policy_action)
    #quasi_policy_action = muopt[tuple([initial_machine_state.T])+tuple(initial_inventory_state.T)]

    phi = evaluate_basis_functions(machine_state, inventory_state, random_action)
 
    machine_state, inventory_state, cost = sample_transitions(
        machine_state,
        inventory_state,
        random_action,
    )

    policy_action = evalute_policy(w,machine_state, inventory_state)
    #policy_action = muopt[tuple([final_machine_state.T])+tuple(final_inventory_state.T)]
    Pphi = evaluate_basis_functions(machine_state, inventory_state, policy_action)

    phidagger = torch.linalg.pinv(phi)
    A = phidagger@phi-gamma*phidagger@Pphi
    b = phidagger@cost
    #w = 0.9*w+0.1*(torch.linalg.pinv(A) @ b)
    w = torch.linalg.pinv(A) @ b
    #A = phi.T @ (phi - gamma * Pphi)
    #b = (phi.T @ cost)
    #u,s,vt = torch.linalg.svd(A)
    # print(torch.allclose(u@torch.diag(s)@vt,A))
    #w = torch.linalg.solve(torch.diag(s)@vt,u.T@b).unsqueeze(-1)
    #w = (torch.linalg.pinv(A) @ b).unsqueeze(-1)

plt.subplot(2,3,1)
plt.matshow(muopt[0],vmin=0,vmax=2,fignum=False)
plt.subplot(2,3,2)
plt.matshow(muopt[1],vmin=0,vmax=2,fignum=False)
plt.subplot(2,3,3)
plt.matshow(muopt[2],vmin=0,vmax=2,fignum=False)
plt.subplot(2,3,4)
plt.matshow(get_policy(0,w),vmin=0,vmax=2,fignum=False)
plt.subplot(2,3,5)
plt.matshow(get_policy(1,w),vmin=0,vmax=2,fignum=False)
plt.subplot(2,3,6)
plt.matshow(get_policy(2,w),vmin=0,vmax=2,fignum=False)
plt.show()