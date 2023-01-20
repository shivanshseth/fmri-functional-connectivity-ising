from scripts.ising_simulation import IsingSimulation
import numpy as np
from joblib import Parallel, delayed
from matplotlib import pyplot as plt


n_jobs = 20
def calc_susceptibility(beta, J, n_rois, eq_steps=300, mc_steps=200):
    sim = IsingSimulation(n_rois, beta, coupling_mat = True, J=J)
    energy_eq = []
    mag_eq = []
    for i in range(eq_steps):
        sim.step()
        energy_eq.append(sim.calcEnergy())
        mag_eq.append(sim.calcMag())
    mag = []
    mag_squared = []
    energy = []
    energy_squared = []
    for i in range(mc_steps):
        sim.step()
        mag.append(sim.calcMag())
        energy.append(sim.calcEnergy())
        mag_squared.append(mag[-1]**2)
        energy_squared.append(energy[-1]**2)
    chi = (np.mean(mag_squared) - np.mean(mag)**2) / (beta * n_rois * n_rois) 
    # return chi, mag, energy, energy_eq, mag_eq
    return chi

ising_reps = np.load('../data/AAL_reps/ising_NYU.npy')
diag = np.load('../data/AAL_reps/diag_NYU.npy')
betas = np.load('../data/AAL_reps/betas_NYU.npy')
n_rois = ising_reps.shape[1]

ising_pos = ising_reps[diag == 1]
ising_neg = ising_reps[diag == 0]

beta_range = np.arange(0.01, 0.1, 0.0005)
chi_pos = []
for beta in beta_range:
    chi_pos.append(Parallel(n_jobs=20)(delayed(calc_susceptibility)(beta, J, n_rois) for J in ising_pos))

chi_neg = []
for beta in beta_range:
    chi_neg.append(Parallel(n_jobs=6)(delayed(calc_susceptibility)(beta, J, n_rois) for J in ising_neg))

chi_pos = np.array(chi_pos)
chi_neg = np.array(chi_neg)

np.save('../data/AAL_reps/chi_pos_NYU.npy', chi_pos)
np.save('../data/AAL_reps/chi_neg_NYU.npy', chi_neg)