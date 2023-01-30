import numpy as np
from numpy.random import rand

class IsingSimulation:
    
    def __init__(self, n_rois, beta, coupling_mat = False, J=None, initial_state = False, state=None):
        self.N = n_rois
        self.beta = beta
        self.costs = []
        if not coupling_mat:
            J = np.random.uniform(0, 1, size=(n_rois, n_rois))
            J = (J + J.T)/2 # making it symmetric
            np.fill_diagonal(J, 1)
        self.J = J
        if not initial_state:
            self.state = 2*np.random.randint(2, size=(n_rois))-1
        else:
            self.state = state
        self.flips = {
            'r': [],
            'e': []
        }
        self.M = []
        self.E = []
        return


    def step(self, update_state = True, state = None):
        if update_state:
            state = self.state[:]
        energy_flips = 0
        random_flips = 0
        # for i in range(self.N):
            # calculating delH
        i = np.random.randint(self.N)
        H_i = 0
        H_i = self.J[i, :] @ state.T 
        H_i -= self.J[i, i] * state[i] # removing self coupling term
        cost =  2 * state[i] * H_i
        if cost < 0:
            state[i] *= -1
            energy_flips +=1
        elif rand() < np.exp(-cost*self.beta):
            self.costs.append(np.exp(-cost*self.beta))
            state[i] *= -1
            random_flips += 1
        if update_state:
            self.state = state
        self.flips['r'].append(random_flips)
        self.flips['e'].append(energy_flips)
        return state
    
    def calcEnergy(self):
        H = 0
        H = -self.state @ self.J @ self.state.T
        return H/2
    
    def calcMag(self):
        mag = np.sum(self.state)
        return mag
    
    def getTimeseries(self, n_timesteps):
        time_series = np.zeros((n_timesteps, self.N))
        state = self.state[:]
        for i in range(n_timesteps):
            state = self.step(False, state)
            time_series[i] = state
        # print(np.sum(time_series))
        fc = 1/n_timesteps * time_series.T @ time_series 
        return time_series, fc
    
    def run(self, steps, n_timesteps, calc_corr=False, fc=None, eqSteps=1000):
        E = M = corr = []
        for i in range(eqSteps):
            sim.step()
            
        for i in range(steps):
            if calc_corr and i%calc_corr == 0:
                time_series, sim_fc = sim.getTimeseries(n_timesteps)
                c = np.corrcoef(np.triu(fc).flatten(), np.triu(sim_fc).flatten())[0, 1]
                corr.append(c)
            sim.step()           # Monte Carlo moves
            E.append(sim.calcEnergy())
            M.append(sim.calcMag())
        if calc_corr:
            return E, M, corr
        return E, M