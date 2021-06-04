import itertools
import copy
import numpy as np
import functools
import time
import torch
import matplotlib.pyplot as plt


class SingleProductDemand():
    def __init__(self):
        self.demand = None
        self.probs = None
        self.max_demand = None
        self.torch_distrib = None

    def iterate(self):
        return zip(self.demand, self.probs)

    def sample(self, num_samples):
        return self.torch_distrib.sample((num_samples,))

    @classmethod
    def from_prob_array(cls, probabilities):
        _d = cls()
        _d.max_demand = len(probabilities)-1
        _d.probs = [p/sum(probabilities) for p in probabilities]
        _d.demand = list(range(_d.max_demand+1))
        _d.torch_distrib = torch.distributions.Categorical(
            torch.tensor(probabilities))
        return _d


class Demands():
    def __init__(self, demands):
        self.demands = demands
        self.w = torch.tensor([d.probs for d in demands], dtype = torch.float)

    def __iter__(self):
        iterable = itertools.product(*[d.iterate() for d in self.demands])
        iterable = map(lambda l:
                       (
                           np.asarray([x[0] for x in l], dtype=np.int32),
                           [x[1] for x in l]
                       ), iterable)
        return iterable

    def single_iter(self, i):
        return self.demands[i].iterate()

    def sample(self, num_samples):
        return torch.stack([d.sample(num_samples) for d in self.demands])

    @classmethod
    def from_repeated_demand(cls, demand, N):
        _d = cls([copy.deepcopy(demand) for _ in range(N)])
        return _d


class State():
    def __init__(self):
        self.N = None
        self.max_invs = None

    def __iter__(self):
        iterable = itertools.product(*[range(m+1) for m in self.max_invs])
        as_np_array = map(np.asarray, iterable)
        return as_np_array

    def single_iter(self, i):
        return range(self.max_invs[i]+1)

    @classmethod
    def from_max_invs(cls, max_invs):
        _s = cls()
        _s.N = len(max_invs)
        _s.max_invs = max_invs
        return _s


class Problem():
    def __init__(self, demands, states, makes):
        # assert abs(sum([np.prod([x[1] for x in d])
        #           for d in demands.iterate()]) - 1) < 1e-10
        assert len(demands.demands) == states.N
        self.demands = demands
        self.states = states
        self.makes = np.asarray(makes)
        self.num_item = self.states.N
        self.num_actions = self.num_item+1
        self.inv_dimensions = [d+1 for d in self.states.max_invs]

    def dynamic(self, inventory_state, machine_state, action, demand):
        cost = 0.0
        if action > 0:
            item_to_prod = action-1
            if action == machine_state:
                inventory_state[item_to_prod] += self.makes[item_to_prod]
            elif action % 2 == machine_state % 2:
                inventory_state[item_to_prod] += self.makes[item_to_prod]//2
                cost += 40
            else:
                inventory_state[item_to_prod] += self.makes[item_to_prod]//4
                cost += 80
        tmp_state = inventory_state-demand
        unsatisfied_demand = -tmp_state[tmp_state < 0]
        cost += 100*unsatisfied_demand.sum().item()
        tmp_state[tmp_state < 0] = 0
        # TODO: maybe switch order next two lines
        tmp_state = np.minimum(tmp_state, self.states.max_invs)
        cost += tmp_state.sum().item()
        return cost, tmp_state, action

    def single_dynamic(self, item_to_prod, initial_inventory_state, action, demand):
        inv_state = initial_inventory_state
        cost = 0

        if action == 0: # 0 - Do not produce
            pass
        elif action == 1:  # 1 - startup from different familiy
            inv_state += self.makes[item_to_prod]//4
            cost += 80
        elif action == 2: # 2 - startup from same familiy
            inv_state += self.makes[item_to_prod]//2
            cost += 40
        elif action == 3: # 3 - no startup
            inv_state += self.makes[item_to_prod]
        else:
            raise NotImplementedError()

        inv_tmp_state = inv_state-demand
        unsatisfied_demand = -min(inv_tmp_state, 0)
        cost += 100*unsatisfied_demand
        inv_tmp_state = min(max(inv_tmp_state, 0),
                            self.states.max_invs[item_to_prod])
        cost += inv_tmp_state
        return cost, inv_tmp_state

    def generate_Pi_and_R_by_decomposition(self):
        matrices = []
        rewards = []
        for item in range(self.num_item):
            matrix = np.zeros([4]+[self.inv_dimensions[item]]*2)
            reward = np.zeros([4]+[self.inv_dimensions[item]])
            for initial_inventory_state in self.states.single_iter(item):
                for (demand, prob) in self.demands.single_iter(item):
                    # 0 - Do not produce
                    # 1 - startup from different familiy
                    # 2 - startup from same familiy
                    # 3 - no startup
                    for action in [0, 1, 2, 3]:
                        cost, next_state = self.single_dynamic(
                            item, initial_inventory_state, action, demand
                        )
                        matrix[action, initial_inventory_state,
                               next_state] += prob
                        reward[action, initial_inventory_state] += prob*cost
            matrices.append(matrix)
            rewards.append(reward)

        return matrices, rewards

    def get_single_action(self, global_action, machine_state, item_idx):
        if global_action != item_idx+1:                 #non sto per produrre item_idx
            return 0
        elif global_action == machine_state:            #sto continuando a produrre item_idx
            return 3
        elif global_action % 2 == machine_state % 2:    #prodotto precedente della stessa famiglia
            return 2
        else:                                           #prodotto precedente di famiglia diversa
            return 1

    def value_iteration(self, gamma=0.99, device='cuda'):
        V = torch.zeros([self.num_actions]+self.inv_dimensions).to(device)
        Q = torch.rand([self.num_actions]*2+self.inv_dimensions).to(device)
        Ms, Rs = self.generate_Pi_and_R_by_decomposition()
        Ms = [torch.from_numpy(m).float().to(device) for m in Ms]
        Rs = [torch.from_numpy(r).float().to(device) for r in Rs]
        print('generated')
        start = time.time()
        for iteration in range(200000):
            # print(iteration)
            for machine_state in range(self.num_actions):
                for action in range(self.num_actions):
                    selectedMs = [Ms[i][self.get_single_action(action, machine_state, i)]
                                      for i in range(self.num_item)]
                    selectedRs = [Rs[i][self.get_single_action(action, machine_state, i)]
                                      for i in range(self.num_item)]

                    #Vnew = gamma*np.einsum('ABCD,iA,jB,kC,lD->ijkl',V,*selectedMs,optimize='optimal')
                    Vnew = gamma * \
                        functools.reduce(lambda W, m: torch.tensordot(
                            W, m, dims=([0], [1])), selectedMs, V[action])

                    for i in range(self.num_item):
                        Vnew += selectedRs[i].reshape([1]
                                                    * i+[-1]+[1]*(self.num_item-i-1))

                    Q[action, machine_state] = Vnew
            Vnew = Q.min(0)[0]

            if (V-Vnew).abs().max().item() < 1e-15:  # torch.allclose(V, Vnew):
                print(f'Converged at iteration {iteration}!')
                break
            V = Vnew
        else:
            print('Did not converge :(')
        print(f'Elapsed time decoupled: {time.time()-start}')
        return Vnew, Q

    def generic_iteration_directly_decomposed_pytorch(self, gamma=0.99, device='cuda'):
        # V(machinestate, invetorystate)
        V = torch.rand([self.num_actions]+self.inv_dimensions, device=device)
        # mu(machinestate, invetorystate)
        mu = torch.zeros_like(V, dtype=torch.int64, device=device)
        # Q(action, machinestate, inventorystate)
        Q = torch.rand([self.num_actions]*2+self.inv_dimensions, device=device)

        Ms, Rs = self.generate_Pi_and_R_by_decomposition()
        Ms = [torch.from_numpy(m).float().to(device) for m in Ms]
        Rs = [torch.from_numpy(r).float().to(device) for r in Rs]
        print('generated')
        start = time.time()
        for iteration in range(20):
            print(iteration)
            for k in range(max(5,iteration**2//5)):
                for machine_state in range(self.num_actions):
                    for action in range(self.num_actions):
                        selectedMs = [Ms[i][self.get_single_action(action, machine_state, i)]
                                      for i in range(self.num_item)]
                        selectedRs = [Rs[i][self.get_single_action(action, machine_state, i)]
                                      for i in range(self.num_item)]

                        Vnew = gamma * \
                            functools.reduce(lambda W, m: torch.tensordot(
                                W, m, dims=([0], [1])), selectedMs, V[action])

                        for i in range(self.num_item):
                            Vnew += selectedRs[i].reshape([1]
                                                          * i+[-1]+[1]*(self.num_item-i-1))

                        Q[action, machine_state] = Vnew

                V = torch.gather(Q, 0, mu.unsqueeze(0).repeat(
                    tuple([self.num_actions]+[1]*len(mu.shape))))[0]
            munew = Q.argmin(0)
            if torch.all(munew == mu):
                print(f'Converged at iteration {iteration}!')
                break
            mu = munew
        else:
            print('Did not converge')
        print(f'Elapsed time decoupled: {time.time()-start}')
        return V, Q

    def simulate(self, initial_inventory_state, initial_machine_state, policy, time_steps = 500, repetitions = 100, gamma = 0.99):
        policy = policy.cpu().numpy()
        total_costs = np.zeros(repetitions)
        for r in range(repetitions):
            inventory_state = np.array(initial_inventory_state)
            machine_state = initial_machine_state
            for t in range(time_steps):
                action = policy[tuple([machine_state])+tuple(inventory_state)]
                demands = self.demands.sample(1).flatten().cpu().numpy()
                cost, inventory_state, machine_state = self.dynamic(inventory_state, machine_state, action, demands)
                machine_state = machine_state.item()
                total_costs[r] += cost * (gamma**t)
        return total_costs

    @classmethod
    def from_single(cls, items_num, demand_probs, max_inv, make):
        _d = SingleProductDemand.from_prob_array(demand_probs)
        _demands = Demands.from_repeated_demand(_d, items_num)
        _states = State.from_max_invs([max_inv]*items_num)
        return cls(_demands, _states, [make]*items_num)

    @classmethod
    def from_multiple(cls, demand_probs, max_inv, make):
        _demands = Demands([SingleProductDemand.from_prob_array(d)
                           for d in demand_probs])
        _states = State.from_max_invs([max_inv]*len(demand_probs))
        return cls(_demands, _states, [make]*len(demand_probs))

