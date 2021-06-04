import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.linalg

production_makes = torch.tensor([[0,0],[12,0],[0,12]])
production_setup = torch.tensor([[0,0],[6,0],[0,6]])
setup_cost = 40.0
demandsDistributions = torch.distributions.Categorical(torch.tensor([[1,2,3,4,5,6],[6,5,4,3,2,1]]))
max_inv = 40
L = 1000
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
    return action, inv_state, cost

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
                inv_state[:,0]/(inv_state[:,1]+1),inv_state[:,1]/(inv_state[:,0]+1),
                inv_state[:,0]**2, inv_state[:,1]**2, inv_state[:,0]*inv_state[:,1],
                torch.exp(-torch.maximum(inv_state[:,0],inv_state[:,1])/10),
                #inv_state[:,0]**3, inv_state[:,1]**3, inv_state[:,0]**2*inv_state[:,1],inv_state[:,0]*inv_state[:,1]**2,
                torch.exp(-(inv_state[:,0]**2+10*inv_state[:,1])/50),
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

w = torch.rand((99,))
for k in range(500):
    if k % 100 == 0:
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

    cost /= 100

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

muopt = torch.LongTensor(np.load('mu.npy'))
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