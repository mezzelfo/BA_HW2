import matplotlib.pyplot as plt
import torch
from torch._C import device
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import numpy as np

production_makes = torch.tensor([[0,0],[12,0],[0,12]],device='cuda')
production_setup = torch.tensor([[0,0],[6,0],[0,6]],device='cuda')
setup_cost = 40.0
demandsDistributions = torch.distributions.Categorical(torch.tensor([[1,2,3,4,5,6],[6,5,4,3,2,1]], device='cuda'))
max_inv = 40
L = 1000
gamma = 0.99
N = 2

def sample_transitions(initial_machine_state, initial_inventory_state, action):
    # initial_machine_state:    tensor of size L
    # initial_inventory_state:  tensor of size LxN
    # action:                   tensor of size L
    #demds = demandsDistributions.sample((L*(N+1),)) #tensor of size LxN
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
        y = F.leaky_relu(self.fc1(y), negative_slope=0.2)
        y = F.leaky_relu(self.fc2(y), negative_slope=0.2)
        y = F.leaky_relu(self.fc3(y), negative_slope=0.2)
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
#optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

losses = []
for epoch in range(100):
    for tick in range(max(10,epoch)):
        m_init = torch.randint(0,N+1,(L,), device='cuda')
        i_init = torch.randint(0,max_inv+1,(L,N), device='cuda')
        a_init = torch.randint(0,N+1,(L,), device='cuda')

        m, i, c = sample_transitions(m_init, i_init, a_init)
        c /= 1000
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
plt.plot(losses[5:,0], '-o')
plt.plot(losses[5:,0]*0, '--')
plt.subplot(1,2,2)
plt.plot(losses[5:,1], '-o')
plt.plot(losses[5:,1]*0, '--')


inv = torch.arange(0,max_inv+1)
X, Y = torch.meshgrid(inv,inv)
inv_state = torch.vstack([X.ravel(), Y.ravel()]).T.cuda()
m_state = torch.ones((inv_state.shape[0],),dtype=torch.long, device='cuda')

muopt = torch.LongTensor(np.load('mu.npy')).cuda()
Qopt = torch.LongTensor(np.load('Q.npy')).cuda()
with torch.no_grad():
    mu0 = net.get_actions(m_state*0, inv_state).reshape((max_inv+1,max_inv+1))
    mu1 = net.get_actions(m_state*1, inv_state).reshape((max_inv+1,max_inv+1))
    mu2 = net.get_actions(m_state*2, inv_state).reshape((max_inv+1,max_inv+1))
    mu = torch.stack([mu0,mu1,mu2])
    plt.figure(2)
    plt.subplot(2,3,1)
    plt.matshow(muopt[0].cpu(),vmin=0,vmax=2,fignum=False)
    plt.subplot(2,3,2)
    plt.matshow(muopt[1].cpu(),vmin=0,vmax=2,fignum=False)
    plt.subplot(2,3,3)
    plt.matshow(muopt[2].cpu(),vmin=0,vmax=2,fignum=False)
    plt.subplot(2,3,4)
    plt.matshow(mu[0].cpu(),vmin=0,vmax=2,fignum=False)
    plt.subplot(2,3,5)
    plt.matshow(mu[1].cpu(),vmin=0,vmax=2,fignum=False)
    plt.subplot(2,3,6)
    plt.matshow(mu[2].cpu(),vmin=0,vmax=2,fignum=False)
    plt.pause(1)


    plt.figure(3)
    Q_m0 = net.forward(m_state*0, inv_state).reshape((max_inv+1,max_inv+1,N+1)).cpu().numpy()
    Q_m1 = net.forward(m_state*1, inv_state).reshape((max_inv+1,max_inv+1,N+1)).cpu().numpy()
    Q_m2 = net.forward(m_state*2, inv_state).reshape((max_inv+1,max_inv+1,N+1)).cpu().numpy()
    for i,Q_mi in enumerate([Q_m0,Q_m1,Q_m2]):
        for action in range(3):
            plt.subplot(3,3,i*3+action+1)
            plt.contour(Q_mi[:,:,action],fignum=False,levels = 50)

    plt.figure(4)
    for i in range(3):
        for action in range(3):
            plt.subplot(3,3,i*3+action+1)
            plt.contour(Qopt[action,i,:,:].cpu().numpy(),fignum=False,levels = 50)

plt.show()

# class muNet(nn.Module):
#     def __init__(self):
#         super(muNet, self).__init__()
#         s = 2
#         self.fc1 = nn.Linear(N+(N+1),s*N)
#         self.fc2 = nn.Linear(s*N,s*N)
#         self.fc3 = nn.Linear(s*N,s*N)
#         self.fc4 = nn.Linear(s*N,N+1)

#     def forward(self, machine_states, inventory_states):
#         one_hot_machine_states = F.one_hot(machine_states, N+1)
#         y = torch.hstack([
#             inventory_states,
#             one_hot_machine_states
#             ]).float()
#         y = F.relu(self.fc1(y))
#         y = F.relu(self.fc2(y))
#         y = F.relu(self.fc3(y))
#         y = self.fc4(y)
#         return y

#     def get_action(self, machine_states, inventory_states):
#         return self.forward(machine_states, inventory_states).argmin(1)

# net = muNet().cuda()
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

# losses = []
# for epoch in range(1):
#     with torch.no_grad():
#         m_init = torch.randint(0,N+1,(L,), device='cuda')
#         i_init = torch.randint(0,max_inv+1,(L,N), device='cuda')
        
#         m = m_init.clone().repeat_interleave(N+1)
#         i = i_init.clone().repeat_interleave(N+1,0)
#         a = torch.arange(0,N+1, device='cuda').repeat(L)
#         Q = torch.zeros((L*(N+1),), device='cuda')
    
#         for t in range(20):
#             m, i, c = sample_transitions(m, i, a)
#             eps = torch.rand(a.shape, device='cuda')
#             random = torch.randint(0,N+1,a.shape, device='cuda')
#             a = torch.where(eps < 1/(epoch+1), random , net.get_action(m,i))
#             Q += c * gamma**t

#         target = Q.view((L,N+1)).argmin(1).detach()

#     for k in range(10000):
#         optimizer.zero_grad()
#         loss = criterion(net.forward(m_init,i_init), target)
#         loss.backward()
#         optimizer.step()
#         losses.append(loss.item())

# inv = torch.arange(0,max_inv+1)
# X, Y = torch.meshgrid(inv,inv)
# inv_state = torch.vstack([X.ravel(), Y.ravel()]).T.cuda()
# m_state = torch.ones((inv_state.shape[0],),dtype=torch.long, device='cuda')
# with torch.no_grad():
#     mu0 = net.get_action(m_state*0, inv_state).reshape((max_inv+1,max_inv+1))
#     mu1 = net.get_action(m_state*1, inv_state).reshape((max_inv+1,max_inv+1))
#     mu2 = net.get_action(m_state*2, inv_state).reshape((max_inv+1,max_inv+1))

# plt.subplot(2,1,2)
# plt.plot(losses)
# plt.subplot(2,3,1)
# plt.matshow(mu0.cpu(),vmin=0,vmax=2,fignum=False)
# plt.subplot(2,3,2)
# plt.matshow(mu1.cpu(),vmin=0,vmax=2,fignum=False)
# plt.subplot(2,3,3)
# plt.matshow(mu2.cpu(),vmin=0,vmax=2,fignum=False)
# plt.show()