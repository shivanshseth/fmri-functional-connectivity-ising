import pandas as pd
import os
import numpy as np
from nilearn import datasets
from nilearn import image
from nilearn.input_data import NiftiLabelsMasker
from os.path import join
from nilearn import datasets
from nilearn import image
from nilearn.input_data import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure
from scipy import optimize
from sklearn.model_selection import train_test_split, cross_validate
from sklearn import svm
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error, f1_score, precision_score, recall_score
from joblib import Parallel, delayed
from ising_simulation import IsingSimulation

def pad_along_axis(array: np.ndarray, target_length: int, axis: int = 0):
    pad_size = target_length - array.shape[axis]
    if pad_size <= 0:
        return array

    npad = [(0, 0)] * array.ndim
    npad[axis] = (0, pad_size)

    return np.pad(array, pad_width=npad, mode='constant', constant_values=0)

#multiscale = datasets.fetch_atlas_basc_multiscale_2015()
class Abide():
    
    def __init__(
                    self, 
                    func_files_dir = '../../data/raw', 
                    meta_file = '../../metadata.csv', 
                    sites = 'all', 
                    labels = 'aut',
                    pre_parcellated = True,
                    timeseries_dir = '../../data/',
                    scale='AAL',
                    atlas='AAL'
                ):

        self.pre_parcellated = pre_parcellated
        if pre_parcellated: 
            self.timeseries_dir = timeseries_dir 
            self.sites = sites
            filtered_func_files = []
            self.age_labels = []
            self.sex_labels = []
            self.aut_labels = []
            self.func = []
            self._sub_ids = [] 
            self.meta_data = pd.read_csv(meta_file)
            self.scale = scale
            self.atlas = atlas
            return 

        try:
            func_files = os.listdir(func_files_dir)
        except:
            print('Data not in found')
            return

        try: 
            self.meta_data = pd.read_csv(meta_file)
        except:
            print('Metadata not in found')
            return

        for i in func_files:
            sub_id = int(i[i.find('5'):].split('_')[0])
            meta = self.meta_data[self.meta_data['SUB_ID'] == sub_id]
            if sites == 'all' or meta['SITE_ID'].iloc[0] in sites:
                if 'age' in labels:
                    age = float(meta['AGE_AT_SCAN'].iloc[0])
                if 'sex' in labels:
                    sex = int(meta['SEX'].iloc[0]) # 1 -> male, 2 -> female 
                if 'aut' in labels:
                    aut = int(meta['DX_GROUP'].iloc[0]) # 1 -> autistic, 2 -> control

                if 'age' in labels:
                    self.age_labels.append(age)
                if 'sex' in labels:
                    self.sex_labels.append(sex-1)
                if 'aut' in labels:
                    self.aut_labels.append(aut-1)
                filtered_func_files.append(os.path.join(func_files_dir, i))
                self._sub_ids.append(sub_id)
        self.n_samples = len(filtered_func_files)
        self.scale = scale
        self.atlas = atlas
        self.func = filtered_func_files
    
    def __loss(self, J, s, beta):
        J = np.reshape(J, (self.n_rois, self.n_rois))
        term1 = 0
        term2 = 0
        for t in range(self.n_timesteps):
            C = beta * J @ s[t].T
            term1 += C @ s[t].T
            term2 -= np.sum(np.log(np.exp(C) + np.exp(-C)))
        return -(term1+term2)/self.n_timesteps

    def __gradient(self, J, s, beta):
        J = np.reshape(J, (self.n_rois, self.n_rois))
        grad = np.zeros((self.n_rois, self.n_rois))
        for t in range(self.n_timesteps):
            C = beta * J @ s[t].T
            grad += np.outer(s[t], s[t].T) - np.outer(np.tanh(C).T, s[t])
        grad = grad * beta/self.n_timesteps
        return -grad.flatten()


    def gradient_descent(self, max_iterations,w_init,
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
            grad = np.reshape(grad, (self.n_rois, self.n_rois))
            # print(grad.shape)
            delta_w = -learning_rate*grad
            w = w+delta_w
            f_history.append(obj_func(w,*extra_param))
            w_history.append(w)
            if i%10 == 0 and disp: 
                print(f"iteration: {i} loss: {f_history[-1]} grad: {np.sum(grad)}")
            i+=1
            diff = np.absolute(f_history[-1]-f_history[-2])
        w_max = w_history[f_history.index(min(f_history))]
        return w_max.flatten()

    def beta_optimization(self, bold, bold_bin, beta):
        J = np.random.uniform(0, 1, size=(self.n_rois, self.n_rois))
        J = (J + J.T)/2 # making it symmetric
        np.fill_diagonal(J, 0)
        fc = 1/self.n_timesteps * bold.T @ bold
        # J_max = optimize.fmin_cg(loss, x0=J.flatten(), fprime=gradient, args=(bold, beta))
        J_max = self.gradient_descent(self.iterations, J, self.__loss, self.__gradient, extra_param=(bold_bin, beta) , learning_rate=self.alpha, threshold=0.005, disp=False)
        J_max = np.reshape(J_max, (self.n_rois, self.n_rois))
        sim = IsingSimulation(self.n_rois, beta, coupling_mat = True, J=J_max)
        for i in range(self.eq_steps):
            sim.step()
        _, sim_fc = sim.getTimeseries(self.sim_timesteps)
        corr = np.corrcoef(np.triu(fc).flatten(), np.triu(sim_fc).flatten())[0, 1]
        return J_max, corr
    
    def beta_optimization_wrapper(self, bold):
        bold_bin = np.copy(bold)
        bold_bin[np.where(bold_bin >= 0)] = 1
        bold_bin[np.where(bold_bin < 0)] = -1
        results = Parallel(n_jobs=5)(delayed(self.beta_optimization)(bold, bold_bin, i) for i in self.beta_range)
        Js, corrs = np.array(results).T
        max_idx = corrs.index(max(corrs))
        beta_max = self.beta_range[max_idx]
        J_max = Js[max_idx]
        J_max, beta_max

    def parcellate(self):
        res = []
        atlas = self.atlas
        os.mkdir(join(self.timeseries_dir, self.atlas))
        masker = NiftiLabelsMasker(labels_img=self.atlas,
                                    memory='nilearn_cache')
        print('getting parcellated time series...')
        for i in self.func:
            sub_id = int(i[i.find('5'):].split('_')[0])
            k = masker.fit_transform(i)
            np.save(join(self.timeseries_dir, atlas,+ str(sub_id) + '_timeseries.txt'), k)
            res.append(k)
        print('parcellation done')
        return res

    def get_timeseries(self): 
        excluded_subjects = pd.DataFrame(columns=self.meta_data.columns)
        if not self.pre_parcellated:
            self.parcellate()

        timeseries = []
        IDs_subject = []
        diagnosis = []
        age = []
        sex = []
        atlas = self.atlas
        subject_ids = self.meta_data['SUB_ID']
        timeseries = []
        t_max = 0
        for index, subject_id in enumerate(subject_ids):
            this_pheno = self.meta_data[self.meta_data['SUB_ID'] == subject_id]
            this_timeseries = join(self.timeseries_dir, atlas,
                                str(subject_id) + '_timeseries.txt')
            idx = 0
            if not os.path.exists(this_timeseries):
                excluded_subjects.loc[len(excluded_subjects)] = this_pheno.values.flatten().tolist()
                continue
            if (this_pheno['SITE_ID'].iloc[0] in self.sites or self.sites == 'all'):
                t = np.loadtxt(this_timeseries)
                if t.shape[0] == 0 or t.shape[1] == 0: 
                    excluded_subjects.append(this_pheno)
                    continue
                t_max = max(t_max, t.shape[0])
                timeseries.append(t)
                idx = idx + 1
                IDs_subject.append(subject_id)
                diagnosis.append(this_pheno['DX_GROUP'].values[0])
                age.append(this_pheno['AGE_AT_SCAN'].values[0])
                sex.append(this_pheno['SEX'].values[0])
        
        # Padding to fix anomalous scans
        for i, t in enumerate(timeseries):
            timeseries[i] = pad_along_axis(t, t_max, 0)
        timeseries = np.array(timeseries) 
        excluded_subjects.to_csv(join('../../data/excluded', atlas.replace('/', '-') +'.csv'))    
        self.n_timesteps = timeseries.shape[1]
        self.n_rois = timeseries.shape[2]
        return timeseries, IDs_subject, diagnosis, age, sex
    
    def sFC(self):
        data, ID, diag, age, sex = self.get_timeseries()
        correlation_measure = ConnectivityMeasure(kind='correlation', vectorize=True)
        correlation_matrices = correlation_measure.fit_transform(data)
        diag = np.array(diag)
        return correlation_matrices, ID, diag, age, sex
    
    def ising_optimize_cg(self, bold, beta, J): 
        J_max = optimize.fmin_cg(self.__loss, x0=J.flatten(), fprime=self.__gradient, args=(bold, beta), disp=False)
        return J_max
    
    def ising_optimize_gd(self, bold, beta, J): 
        J_max = self.gradient_descent(self.iterations, J, self.__loss, self.__gradient, extra_param=(bold, beta) , learning_rate=self.alpha, threshold=0.005, disp=False)
        J_max = J_max.reshape(self.n_rois, self.n_rois)
        #if not np.all(np.abs(J_max-J_max.T) < 1e-8):
            #print('not symmetric')

        J_max = J_max[np.triu_indices(J_max.shape[0], k = 1)].flatten()
        return J_max

    def ising_coupling(self, method = "GD", iterations=500, alpha=2, beta_range=np.linspace(0, 0.30, 31), sim_timesteps = -1 , beta = False):
        # setting parameters for GD
        if sim_timesteps == -1: 
            self.sim_timesteps = self.n_timesteps
        else: self.sim_timesteps = sim_timesteps
        self.iterations = iterations
        self.alpha = alpha
        self.beta_range = beta_range
        data, ID, diag, age, sex = self.get_timeseries()
        data_bin = np.copy(data)
        data_bin[np.where(data_bin >= 0)] = 1
        data_bin[np.where(data_bin < 0)] = -1
        J = np.random.uniform(0, 1, size=(self.n_rois, self.n_rois))
        J = (J + J.T)/2 # making it symmetric
        np.fill_diagonal(J, 0)
        beta = 0.1
        reps = []
        betas = []
        diag = np.array(diag) - 1
        # idx = np.where(diag > 0)[0][:2]
        # data = np.vstack((data[:2], data[idx]))
        # diag = np.concatenate((diag[:2], diag[idx]))
        # if method == 'CG':
        #     reps = Parallel(n_jobs=20)(delayed(self.ising_optimize_cg)(i, beta, J) for i in data)
        if beta:
            reps = Parallel(n_jobs=20)(delayed(self.ising_optimize_gd)(i, beta, J) for i in data_bin)
        #for i in data:
        else: 
            reps = []
            betas = []
            for idx, i in enumerate(data):
                print(f'Subject: {idx}')
                J, b = self.beta_optimization_wrapper(i)
                reps.append(J)
                betas.append(b)
        #    J_max = optimize.fmin_cg(self.__loss, x0=J.flatten(), fprime=self.__gradient, args=(i, beta), disp=False)
        #    reps.append(J_max)
        reps = np.array(reps)
        betas = np.array(betas)
        return reps, betas ,ID, diag, age, sex
        

if __name__ == '__main__':
    dataset = Abide(sites='NYU', atlas='AAL', scale='AAL')
    reps, betas, ID, diag, age, sex = dataset.ising_coupling(method='GD')
    np.save('../../data/ising_nyu.npy', reps)
    np.save('../../data/beta_nyu.npy', betas)
    np.save('../../data/diag_nyu.npy', diag)