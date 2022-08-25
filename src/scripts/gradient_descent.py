import numpy as np
import matplotlib.pyplot as plt
from ising_simulation import IsingSimulation
from joblib import Parallel, delayed
from scipy import optimize
from scipy.stats import zscore
import pickle
import os
import glob
import pandas as pd
from abide_dataset import Abide

def loss(J, s, beta):
#     print(s.shape)
    J = np.reshape(J, (n_rois, n_rois))
#     print(J.shape)
    term1 = 0
    term2 = 0
    for t in range(n_timesteps):
        C = beta * J @ s[t].T
        term1 += C @ s[t].T
        term2 -= np.sum(np.log(np.exp(C) + np.exp(-C)))
    return -(term1+term2)/n_timesteps

def gradient(J, s, beta):
    J = np.reshape(J, (n_rois, n_rois))
    grad = np.zeros((n_rois, n_rois))
    for t in range(n_timesteps):
        C = beta * J @ s[t].T
        grad += np.outer(s[t], s[t].T) - np.outer(np.tanh(C).T, s[t])
    grad = grad * beta/n_timesteps
    return -grad.flatten()

dataset = Abide(sites='NYU')
data, ids, diagnosis, age, sex = dataset.get_timeseries('AAL', 'AAL') 
data_bin = data.copy()
data_bin[np.where(data >= 0)] = 1
data_bin[np.where(data < 0)] = -1
n_timesteps = data.shape[1]
n_rois = data.shape[2]
eq_steps = 1000
sim_timesteps = 300

def beta_optimization(data, data_bin, beta):
    J = np.random.uniform(0, 1, size=(n_rois, n_rois))
    J = (J + J.T)/2 # making it symmetric
    np.fill_diagonal(J, 0)
    corrs = np.zeros(data.shape[0])
    for idx, bold in enumerate(data_bin):
        fc = 1/n_timesteps * data[idx].T @ data[idx]
        J_max = optimize.fmin_cg(loss, x0=J.flatten(), fprime=gradient, args=(bold, beta))
        J_max = np.reshape(J_max, (n_rois, n_rois))
        sim = IsingSimulation(n_rois, beta, coupling_mat = True, J=J_max)
        for i in range(eq_steps):
            sim.step()
        _, sim_fc = sim.getTimeseries(sim_timesteps)
        corrs[idx] = np.corrcoef(np.triu(fc).flatten(), np.triu(sim_fc).flatten())[0, 1]
    return np.mean(corrs), np.std(corrs)

betas = np.linspace(0, 2, 50)
results = Parallel(n_jobs=20)(delayed(beta_optimization)(data, data_bin, i) for i in betas)
file = open('../results/beta_optimization.pkl', 'wb')
pickle.dump(results, file)
results = np.array(results)
np.savetxt('../results/beta_optimization', results)
