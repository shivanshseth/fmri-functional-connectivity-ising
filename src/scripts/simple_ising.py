import numpy as np
import matplotlib.pyplot as plt
from numpy.random import rand

class IsingSimulation:
    
    def __init__(self, N, beta, J=1, initial_state = False, state=None):
        self.N = N
        self.beta = beta
        self.J = J
        if not initial_state:
            self.state = 2*np.random.randint(2, size=(N, N))-1
        else:
            self.state = state
        self.flips = {
            'r': [],
            'e': []
        }
        self.M = []
        self.E = []
        return


    def step(self):
        config = self.state[:]
        N = self.N
        for i in range(N):
            for j in range(N):
                a = np.random.randint(0, N)
                b = np.random.randint(0, N)
                s =  config[a, b]
                nb = config[(a+1)%N,b] + config[a,(b+1)%N] + config[(a-1)%N,b] + config[a,(b-1)%N]
                cost = 2*s*nb
                if cost < 0:
                    s *= -1
                elif rand() < np.exp(-cost*self.beta):
                    s *= -1
                config[a, b] = s
        self.state = config
        return config
    
    def calcEnergy(self):
        config = self.state[:]
        N = self.N
        energy = 0 
        for i in range(len(config)):
            for j in range(len(config)):
                S = config[i,j]
                nb = config[(i+1)%N, j] + config[i,(j+1)%N] + config[(i-1)%N, j] + config[i,(j-1)%N]
                energy += -nb*S
        return energy/2.  # to compensate for double-counting
    
    def calcMag(self):
        mag = np.sum(self.state)
        return mag
    
    def getTimeseries(self, n_timesteps):
        time_series = np.zeros((n_timesteps, self.N*self.N))
        state = self.state[:]
        for i in range(n_timesteps):
            state = self.step()
            time_series[i] = state.flatten()
#         print(np.sum(time_series))
        fc = 1/n_timesteps * time_series.T @ time_series 
        return time_series, fc