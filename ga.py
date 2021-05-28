import torch
import numpy as np
from geneticalgorithm import geneticalgorithm as ga
import matplotlib.pyplot as plt

production_makes = torch.tensor([[0,0],[8,0],[0,8]])
production_setup = torch.tensor([[0,0],[4,0],[0,4]])
setup_cost = 50.0
demandsDistributions = torch.distributions.Categorical(torch.tensor([[6,5,4],[4,5,6]]))
max_inv = 26
L = 100000
gamma = 0.9

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

def get_action_pytorch(machine_state, inventory_state, coeffs):
    # coeffs:   tensor of size (N+1) x N with scalars in [0,1]
    inv_state = inventory_state/max_inv
    shortage = coeffs[machine_state,:]
    return torch.where(
        torch.all(inv_state > shortage, 1), 0,
        (inv_state / shortage).argmin(1)+1
        )

def get_loss(coeffs):
    c = torch.from_numpy(coeffs).reshape(2+1,2)
    # c:   tensor of size (N+1) x N with scalars in [0,1]
    machine_state = torch.randint(0,2+1,(L,))
    inventory_state = torch.randint(0,max_inv+1,(L,2))
    running_cost = torch.zeros_like(machine_state, dtype=torch.float)
    for t in range(500):
        actions = get_action_pytorch(machine_state, inventory_state, c)
        machine_state, inventory_state, cost = sample_transitions(machine_state, inventory_state, actions)
        running_cost = gamma*running_cost + cost
        #running_cost += cost
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

def validate_policy(policy):
    machine_state = torch.randint(0,2+1,(L,))
    inventory_state = torch.randint(0,max_inv+1,(L,2))

    running_cost = torch.zeros_like(machine_state, dtype=torch.float)
    for t in range(T):
        actions = policy[tuple([machine_state.T])+tuple(inventory_state.T)]
        machine_state, inventory_state, cost = sample_transitions(machine_state, inventory_state, actions)
        #running_cost = gamma*running_cost + cost
        running_cost += cost

    # print(running_cost.mean(),running_cost.std())
    # plt.hist(running_cost.cpu().numpy(),bins=100)
    return (running_cost.mean()/T)*1/(1-gamma)

def validate_policy_stat_dist(policy):
    Vopt = torch.LongTensor(np.load('V.npy'))
    machine_state = torch.randint(0,2+1,(L,))
    inventory_state = torch.randint(0,max_inv+1,(L,2))

    running_V = torch.zeros_like(machine_state, dtype=torch.float)
    for t in range(T):
        actions = policy[tuple([machine_state.T])+tuple(inventory_state.T)]
        machine_state, inventory_state, _ = sample_transitions(machine_state, inventory_state, actions)
        running_V += Vopt[tuple([machine_state.T])+tuple(inventory_state.T)]

    # print(running_cost.mean(),running_cost.std())
    # plt.hist(running_cost.cpu().numpy(),bins=100)
    return running_V.mean()/T

T = 100

muopt = torch.LongTensor(np.load('mu.npy'))
v1 = validate_policy(muopt).item()
v2 = validate_policy_stat_dist(muopt).item()
print(v1,v2,v1/v2)

exit()

algorithm_param = {'max_num_iteration': 10,\
                   'population_size':100,\
                   'mutation_probability':0.1,\
                   'elit_ratio': 0.1,\
                   'crossover_probability': 0.5,\
                   'parents_portion': 0.3,\
                   'crossover_type':'uniform',\
                   'max_iteration_without_improv':None}

model=ga(function=get_loss,dimension=(2+1)*2,variable_type='real',variable_boundaries=np.array([[0,1]]*((2+1)*2)),algorithm_parameters=algorithm_param)
model.run()
res = model.output_dict['variable']
mu = tabulate_mu_from_coeffs(res)
muopt = torch.LongTensor(np.load('mu.npy'))

print('optimal mu value:',validate_policy(muopt))
print('heuristic mu value:',validate_policy(mu))


plt.subplot(2,3,1)
plt.matshow(muopt[0],vmin=0,vmax=2,fignum=False)
plt.subplot(2,3,2)
plt.matshow(muopt[1],vmin=0,vmax=2,fignum=False)
plt.subplot(2,3,3)
plt.matshow(muopt[2],vmin=0,vmax=2,fignum=False)
plt.subplot(2,3,4)
plt.matshow(mu[0],vmin=0,vmax=2,fignum=False)
plt.subplot(2,3,5)
plt.matshow(mu[1],vmin=0,vmax=2,fignum=False)
plt.subplot(2,3,6)
plt.matshow(mu[2],vmin=0,vmax=2,fignum=False)
plt.show()