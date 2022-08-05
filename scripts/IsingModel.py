import numpy as np
import matplotlib.pyplot as plt
from numpy.random import rand


class IsingSimulation:
    
    def __init__(self, n_rois, beta, coupling_mat = False, J=None):
        self.N = n_rois
        self.beta = beta
        if not coupling_mat:
            J = np.random.uniform(0, 1, size=(n_rois, n_rois))
            J = (J + J.T)/2 # making it symmetric
            np.fill_diagonal(J, 1)
        self.J = J
        self.state = 2*np.random.randint(2, size=(n_rois))-1
        return


    def step(self, update_state = True, state = None):
        if update_state:
            state = self.state[:]
        for i in range(self.N):
            # calculating delH
            H_i = 0
            H_i = self.J[i, :] @ state.T 
            H_i -= self.J[i, i] * state[i] # removing self coupling term
            cost = 2 * H_i
            if cost < 0:
                state[i] *= -1
            elif rand() < np.exp(-cost*self.beta):
                state[i] *= -1
        if update_state:
            self.state = state
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
        fc = 1/n_timesteps * time_series.T @ time_series 
        return time_series, fc