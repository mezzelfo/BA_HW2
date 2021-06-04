import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import numpy as np

# production_makes = torch.diag(torch.tensor([0,16,16,16,16]))[1:,:].T.cuda()
# production_setup_same_family = production_makes/2
# production_setup_different_family = production_makes/4
# production_plan = torch.FloatTensor([0,4,8,16]).cuda()
# setup_cost_plan = torch.FloatTensor([0,80,40,0]).cuda()
# setup_cost_same_family = 40.0
# setup_cost_different_family = 80.0

production_makes = torch.tensor([[0,0,0,0],[16,0,0,0],[0,16,0,0],[0,0,16,0],[0,0,0,16]],device='cuda')
production_setup = torch.tensor([[0,0,0,0],[8,0,0,0],[0,8,0,0],[0,0,8,0],[0,0,0,8]],device='cuda')
# production_makes = torch.tensor([[0,0],[12,0],[0,12]],device='cuda')
# production_setup = torch.tensor([[0,0],[6,0],[0,6]],device='cuda')
setup_cost = 40.0

demandsDistributions = torch.distributions.Categorical(torch.tensor([
        [1,2,3,4,5,6],
        [6,5,4,3,2,1],
        [1,2,3,4,5,0],
        [5,4,3,2,1,0],
        ], device='cuda'))
max_inv = 50
L = 1000
gamma = 0.99
N = 4

print(f'Stiamo samplando {L} stati su {(max_inv**N)*(N+1)}, ovvero {100*L/((max_inv**N)*(N+1))}%')

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

class QNet(nn.Module):
    '''
    Approssimazione di Q(s,*) per * in A
    '''
    def __init__(self):
        super(QNet, self).__init__()
        s = 20
        self.fc1 = nn.Linear(N+(N+1),s*N)
        self.fc2 = nn.Linear(s*N,s*N)
        self.fc3 = nn.Linear(s*N,s*N)
        self.fc4 = nn.Linear(s*N,N+1)
    
    def forward(self, machine_states, inventory_states):
        one_hot_machine_states = F.one_hot(machine_states, N+1)
        y = torch.hstack([
            inventory_states,
            one_hot_machine_states
            ]).float()
        y = torch.sigmoid(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))
        y = torch.sigmoid(self.fc3(y))
        y = self.fc4(y)
        return y

    def get_actions(self, machine_states, inventory_states):
        return self.forward(machine_states, inventory_states).argmin(1)

    def get_V(self, machine_states, inventory_states):
        return self.forward(machine_states, inventory_states).min(1)[0]
        

net = QNet().cuda()
net_old = QNet().cuda()
net_old.load_state_dict(net.state_dict())

criterion = nn.SmoothL1Loss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

losses = []
cMax = 0
for epoch in range(400):
    for tick in range(max(10,epoch)):
        m_init = torch.randint(0,N+1,(L,), device='cuda')
        i_init = torch.randint(0,max_inv+1,(L,N), device='cuda')
        a_init = torch.randint(0,N+1,(L,), device='cuda')

        m, i, c = sample_transitions(m_init, i_init, a_init)
        cMax = max(cMax,c.max().item())
        c /= cMax
        target = (c + gamma*net_old.get_V(m,i)).detach()
  
        optimizer.zero_grad()
        Qhat = net.forward(m_init,i_init).gather(1,a_init.unsqueeze(-1)).squeeze()
        loss = criterion(Qhat, target)
        loss.backward()
        optimizer.step()
        losses.append((loss.item(),Qhat.mean().item()))
        
    net_old.load_state_dict(net.state_dict())
print('done')
plt.figure(1)
losses = np.asarray(losses)
plt.subplot(1,2,1)
plt.plot(np.log(losses[5:,0]), '-o')
plt.subplot(1,2,2)
plt.plot(losses[5:,1], '-o')
plt.plot(losses[5:,1]*0, '--')


# muopt = torch.LongTensor(np.load('mu.npy')).cuda()
# Qopt = torch.LongTensor(np.load('Q.npy')).cuda()
inv = torch.arange(0,max_inv+1)
X, Y = torch.meshgrid(inv,inv)
grosso = 35*torch.ones_like(X)
inv_state = torch.vstack([grosso.ravel(), grosso.ravel(), X.ravel(), Y.ravel()]).T.cuda()
m_state = torch.ones((inv_state.shape[0],),dtype=torch.long, device='cuda')
with torch.no_grad():
    mu0 = net.get_actions(m_state*0, inv_state).reshape((max_inv+1,max_inv+1))
    mu1 = net.get_actions(m_state*1, inv_state).reshape((max_inv+1,max_inv+1))
    mu2 = net.get_actions(m_state*2, inv_state).reshape((max_inv+1,max_inv+1))
    mu3 = net.get_actions(m_state*3, inv_state).reshape((max_inv+1,max_inv+1))
    mu4 = net.get_actions(m_state*4, inv_state).reshape((max_inv+1,max_inv+1))
    plt.figure(2)
    plt.subplot(2,3,1)
    plt.matshow(mu0.cpu(),vmin=0,vmax=4,fignum=False)
    plt.subplot(2,3,2)
    plt.matshow(mu1.cpu(),vmin=0,vmax=4,fignum=False)
    plt.subplot(2,3,3)
    plt.matshow(mu2.cpu(),vmin=0,vmax=4,fignum=False)
    plt.subplot(2,3,4)
    plt.matshow(mu3.cpu(),vmin=0,vmax=4,fignum=False)
    plt.subplot(2,3,5)
    plt.matshow(mu4.cpu(),vmin=0,vmax=4,fignum=False)
plt.show()
