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

def gradient_descent(max_iterations,w_init,
                     obj_func,grad_func,extra_param = (),
                     learning_rate=0.05,momentum=0.8, threshold=0.001, disp=False):
    
    w = w_init
    w_history = [w]
    f_history = [obj_func(w,*extra_param)]
    delta_w = np.zeros(w.shape)
    i = 0
    diff = 1.0e10
    
    while i<max_iterations and diff > threshold:
        grad = grad_func(w,*extra_param)
        # print("from func", grad.shape)
        grad = np.reshape(grad, (n_rois, n_rois))
        # print(grad.shape)
        delta_w = -learning_rate*grad
        w = w+delta_w
        f_history.append(obj_func(w,*extra_param))
        w_history.append(w)
        if i%10 == 0 and disp: 
            print(f"iteration: {i} loss: {f_history[-1]} grad: {np.sum(grad)}")
        i+=1
        diff = np.absolute(f_history[-1]-f_history[-2])
    
    return w_history,f_history

dataset = Abide(sites='NYU', scale='AAL', atlas='AAL')
data, ids, diagnosis, age, sex = dataset.get_timeseries() 
data_bin = data.copy()
data_bin[np.where(data >= 0)] = 1
data_bin[np.where(data < 0)] = -1
n_timesteps = data.shape[1]
n_rois = data.shape[2]
eq_steps = 1000
sim_timesteps = 300
alpha = 10
iterations = 500
def beta_optimization(data, data_bin, beta):
    J = np.random.uniform(0, 1, size=(n_rois, n_rois))
    J = (J + J.T)/2 # making it symmetric
    np.fill_diagonal(J, 0)
    corrs = np.zeros(data.shape[0])
    for idx, bold in enumerate(data_bin):
        fc = 1/n_timesteps * data[idx].T @ data[idx]
        # J_max = optimize.fmin_cg(loss, x0=J.flatten(), fprime=gradient, args=(bold, beta))
        J_hist, f_hist = gradient_descent(iterations, J, loss, gradient, extra_param=(bold, beta) , learning_rate=alpha, threshold=0.005, disp=True)
        J_max = J_hist[f_hist.index(min(f_hist))]
        J_max = np.reshape(J_max, (n_rois, n_rois))
        sim = IsingSimulation(n_rois, beta, coupling_mat = True, J=J_max)
        for i in range(eq_steps):
            sim.step()
        _, sim_fc = sim.getTimeseries(sim_timesteps)
        corrs[idx] = np.corrcoef(np.triu(fc).flatten(), np.triu(sim_fc).flatten())[0, 1]
    return np.mean(corrs), np.std(corrs)

betas = np.linspace(0, 1, 100)
results = Parallel(n_jobs=20)(delayed(beta_optimization)(data, data_bin, i) for i in betas)
#results = beta_optimization(data, data_bin, 0.1)
results = np.array(results)
np.save('../results/beta_optimization_gd_fine', results)
