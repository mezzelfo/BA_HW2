import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import numpy as np

production_makes = torch.tensor([[0,0],[12,0],[0,12]],device='cuda')
production_setup = torch.tensor([[0,0],[6,0],[0,6]],device='cuda')
setup_cost = 40.0
demandsDistributions = torch.distributions.Categorical(torch.tensor([[1,2,3,4,5,6],[6,5,4,3,2,1]], device='cuda'))
max_inv = 20
L = 100
gamma = 0.8
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
        s = 50
        self.fc1 = nn.Linear(N+(N+1),s*N)
        self.fc2 = nn.Linear(s*N,s*N)
        self.fc3 = nn.Linear(s*N,s*N)
        self.fc4 = nn.Linear(s*N+(N+1)+N,N+1)
    
    def forward(self, machine_states, inventory_states):
        one_hot_machine_states = F.one_hot(machine_states, N+1)
        y = torch.hstack([
            1/(inventory_states+1),
            one_hot_machine_states
            ]).float()
        y = F.relu(self.fc1(y))
        y = F.relu(self.fc2(y))
        y = F.relu(self.fc3(y))
        y = torch.hstack([
            y,
            one_hot_machine_states,
            inventory_states
            ])
        y = self.fc4(y)
        return y

    def get_actions(self, machine_states, inventory_states):
        return self.forward(machine_states, inventory_states).argmin(1)

net = QNet().cuda()

criterion =  nn.SmoothL1Loss()
#optimizer =  torch.optim.SGD(net.parameters(), lr = 0.1)
optimizer =  torch.optim.Adam(net.parameters())

#muopt = torch.LongTensor(np.load('mu.npy')).cuda()

losses = []
for epoch in range(5000):
    m = torch.randint(0,N+1,(L,), device='cuda')
    i = torch.randint(0,max_inv+1,(L,N), device='cuda')
    a = torch.randint(0,N+1,(L,), device='cuda')
    
    # a = torch.arange(0,N+1,device='cuda')
    # m_rep = m.repeat_interleave(N+1)
    # i_rep = i.repeat_interleave(N+1,0)
    # a_rep = a.repeat(L)
    # mnext, inext, c = sample_transitions(m_rep,i_rep,a_rep)

    mnext, inext, c = sample_transitions(m,i,a)
    

    Qhat = net.forward(m,i).gather(1, a.unsqueeze(-1)).squeeze()
    Qhatnext = net.forward(mnext,inext).min(1)[0] * gamma + c + c/(1-gamma)*0.99**(epoch/2)
    Qhatnext = Qhatnext.detach()
    loss = ((Qhat-Qhatnext).abs()).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # tot_loss = 0
    # for t in range(1000):
    #     Qhat = net.forward(m,i)
    #     mnext, inext, c = sample_transitions(m,i,a)
    #     Qhatnext = net.forward(mnext,inext).min(1)[0].detach() * gamma + c
    #     Qhatnext = Qhatnext.unsqueeze(-1)
    #     optimizer.zero_grad()
    #     loss = (Qhat-Qhatnext).abs().mean()
    #     optimizer.step()
    #     tot_loss += loss.item()
    #     # Qhat = net.forward(m,i)
    #     # #Q = Qopt[tuple(torch.hstack([a.unsqueeze(-1),m.unsqueeze(-1),i]).T)]
        
    #     # muhat = Qhat.argmin(1)
    #     # epsgreedy = torch.where(torch.rand((L,), device='cuda')<0.2, torch.randint(0,N+1,(L,), device='cuda'), muhat)
    #     # # epsgreedy = muopt[tuple(torch.hstack([m.unsqueeze(-1),i]).T)]
    #     # Qhat = Qhat.gather(1, epsgreedy.unsqueeze(-1)).squeeze()
        
    #     # mnext, inext, c = sample_transitions(m,i,epsgreedy)
    #     # Qhatnext = net.forward(mnext,inext).min(1)[0].detach() * gamma + c
        
    #     # optimizer.zero_grad()
    #     # loss = (Qhat-Qhatnext).abs().mean()#criterion(Qhat, Qhatnext)#(Qhat-Qhatnext).square().mean()#criterion(Qhat, Qhatnext)#+Qhat.mean()/10
    #     # loss.backward()
    #     # optimizer.step()
    #     # tot_loss += loss.item()
    
    losses.append((epoch,loss.item()))
    if epoch % 100 == 0:
        print(epoch,loss.item(), Qhat.mean().item(), Qhatnext.mean().item(), c.mean().item())
    #net_slow.load_state_dict(net.state_dict())
        
print('done')
plt.figure(1)
losses = np.asarray(losses)
plt.plot(losses[5:,0],losses[5:,1], '-o')
    
inv = torch.arange(0,max_inv+1)
X, Y = torch.meshgrid(inv,inv)
inv_state = torch.vstack([X.ravel(), Y.ravel()]).T.cuda()
m_state = torch.ones((inv_state.shape[0],),dtype=torch.long, device='cuda')

#muopt = torch.LongTensor(np.load('mu.npy')).cuda()
with torch.no_grad():
    mu0 = net.get_actions(m_state*0, inv_state).reshape((max_inv+1,max_inv+1))
    mu1 = net.get_actions(m_state*1, inv_state).reshape((max_inv+1,max_inv+1))
    mu2 = net.get_actions(m_state*2, inv_state).reshape((max_inv+1,max_inv+1))
    mu = torch.stack([mu0,mu1,mu2])

    
    plt.figure(2)
    # plt.subplot(2,3,1)
    # plt.matshow(muopt[0].cpu(),vmin=0,vmax=2,fignum=False)
    # plt.subplot(2,3,2)
    # plt.matshow(muopt[1].cpu(),vmin=0,vmax=2,fignum=False)
    # plt.subplot(2,3,3)
    # plt.matshow(muopt[2].cpu(),vmin=0,vmax=2,fignum=False)
    plt.subplot(2,3,4)
    plt.matshow(mu[0].cpu(),vmin=0,vmax=2,fignum=False)
    plt.subplot(2,3,5)
    plt.matshow(mu[1].cpu(),vmin=0,vmax=2,fignum=False)
    plt.subplot(2,3,6)
    plt.matshow(mu[2].cpu(),vmin=0,vmax=2,fignum=False)
    plt.pause(1)

plt.show()
