import itertools
import copy
import numpy as np
import math
import functools
import time
import torch


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
    def __init__(self):
        self.demands = None

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
        _d = cls()
        _d.demands = [copy.deepcopy(demand) for _ in range(N)]
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

    def dynamic(self, initial_state, action, demand):
        state = np.copy(initial_state)
        cost = 0
        if action > 0:
            item_to_prod = action-1
            state[item_to_prod] += self.makes[item_to_prod]
            # TODO: add production cost

        tmp_state = state-demand
        unsatisfied_demand = -tmp_state[tmp_state < 0]
        cost += 100*unsatisfied_demand.sum()
        tmp_state[tmp_state < 0] = 0
        # TODO: maybe switch order next two lines
        tmp_state = np.minimum(tmp_state, self.states.max_invs)
        cost += tmp_state.sum()
        return cost, tmp_state

    def single_dynamic(self, item_to_prod, initial_state, action, demand):
        state = initial_state
        cost = 0
        if action > 0:
            state += self.makes[item_to_prod]
        tmp_state = state-demand
        unsatisfied_demand = -min(tmp_state, 0)
        cost += 100*unsatisfied_demand
        tmp_state = min(max(tmp_state, 0), self.states.max_invs[item_to_prod])
        cost += tmp_state
        return cost, tmp_state

    def generate_Pi_and_R_by_exploration(self):
        matrix = np.zeros([self.num_actions]+self.inv_dimensions*2)
        reward = np.zeros([self.num_actions]+self.inv_dimensions)
        for initial_state in self.states:
            for (demands, probs) in self.demands:
                for action in range(self.num_actions):
                    cost, nextstate = self.dynamic(
                        initial_state, action, demands
                    )
                    all_coord = tuple(np.concatenate(
                        ([action], initial_state, nextstate)))
                    pi = math.prod(probs)
                    matrix[all_coord] += pi
                    reward[all_coord[:(self.num_item+1)]] += pi*cost

        assert all([np.allclose(matrix[i].sum(axis=tuple(
            range(self.num_item, self.num_item*2))), 1) for i in range(self.num_actions)])
        return matrix, reward

    def generate_Pi_and_R_by_decomposition(self):
        matrices = []
        rewards = []
        for item in range(self.num_item):
            matrix = np.zeros([2]+[self.inv_dimensions[item]]*2)
            reward = np.zeros([2]+[self.inv_dimensions[item]])
            for initial_state in self.states.single_iter(item):
                for (demand, prob) in self.demands.single_iter(item):
                    for action in [0, 1]:  # Only two actions
                        cost, next_state = self.single_dynamic(
                            item, initial_state, action, demand
                        )
                        matrix[action, initial_state, next_state] += prob
                        reward[action, initial_state] += prob*cost
            matrices.append(matrix)
            rewards.append(reward)

        return matrices, rewards

    def value_iteration_assembled_decomposed(self, gamma=0.99):
        V = np.zeros(self.inv_dimensions)
        M, R = self.assemble_from_decomposition()
        start = time.time()
        for iteration in range(1000):
            Vnew = (R + gamma*np.tensordot(M, V, self.num_item)).min(0)
            if np.allclose(V, Vnew):
                print(f'Converged at iteration {iteration}!')
                break
            V = Vnew
        else:
            print('Did not converge :(')
        print(f'Elapsed time assembled: {time.time()-start}')
        #np.savetxt('V.txt', Vnew, delimiter=',')
        return Vnew

    def value_iteration_directly_decomposed(self, gamma=0.99):
        V = np.zeros(self.inv_dimensions, dtype=np.float32)
        Ms, Rs = self.generate_Pi_and_R_by_decomposition()
        Ms = [m.astype(np.float32) for m in Ms]
        Rs = [r.astype(np.float32) for r in Rs]
        print('generated')
        start = time.time()
        for iteration in range(1000):
            # print(iteration)
            vs = []
            for action in range(self.num_actions):
                selectedMs = [Ms[i][1 if i == action-1 else 0]
                              for i in range(self.num_item)]
                selectedRs = [Rs[i][1 if i == action-1 else 0]
                              for i in range(self.num_item)]

                #Vnew = gamma*np.einsum('ABCD,iA,jB,kC,lD->ijkl',V,*selectedMs,optimize='optimal')
                Vnew = gamma * \
                    functools.reduce(lambda W, m: np.tensordot(
                        W, m, axes=([0], [1])), selectedMs, V)

                for i in range(self.num_item):
                    Vnew += selectedRs[i].reshape([1]
                                                  * i+[-1]+[1]*(self.num_item-i-1))

                vs.append(Vnew)
            Vnew = functools.reduce(np.minimum, vs)  # np.asarray(vs).min(0)
            if np.allclose(V, Vnew):
                print(f'Converged at iteration {iteration}!')
                break
            V = Vnew
        else:
            print('Did not converge :(')
        print(f'Elapsed time decoupled: {time.time()-start}')
        #np.savetxt('V2.txt', Vnew, delimiter=',')
        return Vnew

    def value_iteration_directly_decomposed_pytorch(self, gamma=0.99, device = 'cuda'):
        V = torch.zeros(self.inv_dimensions).to(device)
        Ms, Rs = self.generate_Pi_and_R_by_decomposition()
        Ms = [torch.from_numpy(m).float().to(device) for m in Ms]
        Rs = [torch.from_numpy(r).float().to(device) for r in Rs]
        print('generated')
        start = time.time()
        for iteration in range(1000):
            print(iteration)
            vs = []
            for action in range(self.num_actions):
                selectedMs = [Ms[i][1 if i == action-1 else 0]
                              for i in range(self.num_item)]
                selectedRs = [Rs[i][1 if i == action-1 else 0]
                              for i in range(self.num_item)]

                #Vnew = gamma*np.einsum('ABCD,iA,jB,kC,lD->ijkl',V,*selectedMs,optimize='optimal')
                Vnew = gamma * \
                    functools.reduce(lambda W, m: torch.tensordot(
                        W, m, dims=([0], [1])), selectedMs, V)

                for i in range(self.num_item):
                    Vnew += selectedRs[i].reshape([1]
                                                  * i+[-1]+[1]*(self.num_item-i-1))

                vs.append(Vnew)
            Vnew = functools.reduce(torch.minimum, vs)  # np.asarray(vs).min(0)
            if torch.allclose(V, Vnew):
                print(f'Converged at iteration {iteration}!')
                break
            V = Vnew
        else:
            print('Did not converge :(')
        print(f'Elapsed time decoupled: {time.time()-start}')
        #np.savetxt('V2.txt', Vnew, delimiter=',')
        return Vnew

    def assemble_from_decomposition(self):
        matrix = np.zeros([self.num_actions]+self.inv_dimensions*2)
        reward = np.zeros([self.num_actions]+self.inv_dimensions)
        Ms, Rs = self.generate_Pi_and_R_by_decomposition()

        correct_axes = sorted(range(2*self.num_item), key=lambda x: x % 2)

        for action in range(self.num_actions):
            single_actions = [0 for i in range(self.num_item)]
            if action > 0:
                single_actions[action-1] = 1

            # Transition Probabilities Check
            selected = [Ms[i][single_actions[i]] for i in range(self.num_item)]
            #final = np.einsum('ab,zw->azbw',*selected)
            expanded = functools.reduce(
                # lambda x, y: np.tensordot(x, y, 0), selected)
                lambda x, y: np.multiply.outer(x, y), selected)
            expanded = expanded.transpose(*correct_axes)

            matrix[action] = expanded

            # Immediate Reward Check
            #selected = Rs[0][0,:,np.newaxis]+Rs[0][0,np.newaxis,:]
            # np.add.outer(Rs[0][0],Rs[0][0])
            selected = [Rs[i][single_actions[i]] for i in range(self.num_item)]
            expanded = functools.reduce(
                lambda x, y: np.add.outer(x, y), selected
            )

            reward[action] = expanded

        return matrix, reward

    def check_big_from_decomposition(self):
        Me, Re = self.generate_Pi_and_R_by_exploration()
        Md, Rd = self.assemble_from_decomposition()

        assert np.allclose(Me, Md)
        assert np.allclose(Re, Rd)

    def simulate(self, time_steps, repetitions, initial_state):
        raise NotImplementedError()

    @classmethod
    def from_single(cls, items_num, demand_probs, max_inv, make):
        _d = SingleProductDemand.from_prob_array(demand_probs)
        _demands = Demands.from_repeated_demand(_d, items_num)
        _states = State.from_max_invs([max_inv]*items_num)
        return cls(_demands, _states, [make]*items_num)


problem = Problem.from_single(
    items_num=4,
    demand_probs=[1, 1, 1],
    max_inv=100,
    make=2
)
#problem.simulate(50, 500, (0, 0, 0, 0))
# problem.check_big_from_decomposition()

#V1 = problem.value_iteration_assembled_decomposed()
V1 = problem.value_iteration_directly_decomposed_pytorch(device='cuda').cpu()
# V2 = problem.value_iteration_directly_decomposed_pytorch(device='cpu')
# V3 = problem.value_iteration_directly_decomposed()

# print('V1,V2',np.allclose(V1,V2))
# print('V1,V3',np.allclose(V1,V3))
# print('V2,V3',np.allclose(V2,V3))

# np.save('V1.npy',V1)
# np.save('V2.npy',V2)
# np.save('V3.npy',V3)
#assert np.allclose(V1,V2)
