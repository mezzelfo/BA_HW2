import itertools
import copy
import numpy as np
import math
import functools
import time
import torch
import matplotlib.pyplot as plt


class SingleProductDemand():
    def __init__(self):
        self.demand = None
        self.probs = None
        self.max_demand = None

    def iterate(self):
        return zip(self.demand, self.probs)

    @classmethod
    def from_prob_array(cls, probabilities):
        _d = cls()
        _d.max_demand = len(probabilities)-1
        _d.probs = [p/sum(probabilities) for p in probabilities]
        _d.demand = list(range(_d.max_demand+1))
        return _d


class Demands():
    def __init__(self, demands):
        self.demands = demands

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

    def single_dynamic(self, item_to_prod, initial_inventory_state, action, demand):
        inv_state = initial_inventory_state
        cost = 0

        if action == 0: #NON FARE
            pass
        elif action == 1: #PRODUCI NORMALE
            inv_state += self.makes[item_to_prod]
        elif action == 2: #PRODUCI SETUP
            inv_state += self.makes[item_to_prod]//2
            #inv_state += self.makes[item_to_prod]
            cost += 400
        else:
            raise NotImplementedError()

        inv_tmp_state = inv_state-demand
        unsatisfied_demand = -min(inv_tmp_state, 0)
        cost += 100*unsatisfied_demand
        inv_tmp_state = min(max(inv_tmp_state, 0),
                            self.states.max_invs[item_to_prod])
        cost += inv_tmp_state/100
        return cost, inv_tmp_state

    def generate_Pi_and_R_by_decomposition(self):
        matrices = []
        rewards = []
        for item in range(self.num_item):
            matrix = np.zeros([3]+[self.inv_dimensions[item]]*2)
            reward = np.zeros([3]+[self.inv_dimensions[item]])
            for initial_inventory_state in self.states.single_iter(item):
                for (demand, prob) in self.demands.single_iter(item):
                    # Do not produce, Produce normally, produce at startup
                    for action in [0, 1, 2]:
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
        if global_action != item_idx+1:
            return 0
        else:
            if global_action == machine_state:
                return 1
            else:
                return 2

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
        for iteration in range(300):
            print(iteration)
            for k in range(max(10,iteration//3)):
                for machine_state in range(self.num_actions):
                    for action in range(self.num_actions):
                        selectedMs = [Ms[i][self.get_single_action(action, machine_state, i)]
                                      for i in range(self.num_item)]
                        selectedRs = [Rs[i][self.get_single_action(action, machine_state, i)]
                                      for i in range(self.num_item)]

                        Vnew = gamma * \
                            functools.reduce(lambda W, m: torch.tensordot(
                                W, m, dims=([0], [1])), selectedMs, V[machine_state])

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

    def simulate(self, time_steps, repetitions, initial_inventory_state):
        raise NotImplementedError()

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


# problem = Problem.from_multiple(
#     demand_probs=[
#         [1, 2, 3, 4, 5, 4, 3, 2, 1],
#         [5, 4, 3, 2, 1],
#     ],
#     max_inv=100,
#     make=9
# )

problem = Problem.from_single(
    items_num=4,
    demand_probs=[1,2,3,4,5,4,3,2,1],
    max_inv=50,
    make=24
)

V, Q = problem.generic_iteration_directly_decomposed_pytorch()

torch.save(V, 'V.pt')
torch.save(Q, 'Q.pt')
