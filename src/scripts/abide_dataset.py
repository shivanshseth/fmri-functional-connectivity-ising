import pandas as pd
import os
import numpy as np
import pickle as pkl
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

BASE_DIR = "/home/anirudh/Research/Brain/Datasets/mica-mics-dataset-aparca2009s"
# BASE_DIR = "/home/anirudh.palutla/Brain/Datasets/mica-mics-dataset-aparca2009s"
SAVE_DIR = "../../data/mica-mics"
SUBJECTS_SAVE_DIR = os.path.join(SAVE_DIR, "subjects")
SAVE_BY_SUBJECTS = True
EXTENDED_SAVE_FP = os.path.join(SAVE_DIR, "subjects.pkl")
SAVE_EXTENDED = True

FUNC_DIR = os.path.join(BASE_DIR, "timeseries")
META_FP = os.path.join(BASE_DIR, "metadata.csv")
TIMESERIES_DIR = os.path.join(BASE_DIR, "timeseries")
STRUCT_CONN_DIR = os.path.join(BASE_DIR, "sc")
# STRUCT_CONN_DIR = None

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)
if not os.path.exists(SUBJECTS_SAVE_DIR):
    os.makedirs(SUBJECTS_SAVE_DIR)

def pad_along_axis(array: np.ndarray, target_length: int, axis: int = 0):
    pad_size = target_length - array.shape[axis]
    if pad_size <= 0:
        return array

    npad = [(0, 0)] * array.ndim
    npad[axis] = (0, pad_size)

    return np.pad(array, pad_width=npad, mode='constant', constant_values=0)

def is_none_array(arr):
    if (type(arr) != type(None)) and (type(arr[0]) != type(None)):
        return False
    return True

#multiscale = datasets.fetch_atlas_basc_multiscale_2015()
class Abide():
    
    def __init__(
                    self, 
                    func_files_dir = FUNC_DIR, 
                    meta_file = META_FP, 
                    timeseries_dir = TIMESERIES_DIR,
                    struct_conn_dir = STRUCT_CONN_DIR,
                    sites = 'all', 
                    labels = [ 'age', 'sex', 'aut' ],
                    pre_parcellated = True,
                    atlas='AAL',
                ):

        self.pre_parcellated = pre_parcellated
        if pre_parcellated: 
            self.timeseries_dir = timeseries_dir 
            self.struct_conn_dir = struct_conn_dir
            self.sites = sites
            filtered_func_files = []
            self.age_labels = []
            self.sex_labels = []
            self.aut_labels = []
            func_files = os.listdir(join(timeseries_dir, atlas))
            self.func = func_files
            self._sub_ids = [] 
            self.meta_data = pd.read_csv(meta_file)
            self.atlas = atlas
            self.sFC_result = None
            self.timeseries_result = None
            self.SC_data = None

            self.get_timeseries()
            self.SC()
            self.sFC()
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
        self.atlas = atlas
        self.func = filtered_func_files 

    def __loss(self, J, s, beta, sc=None, lambda_=None):
        J = np.reshape(J, (self.n_rois, self.n_rois))
        term1 = 0
        term2 = 0
        term3 = 0
        for t in range(self.n_timesteps):
            C = beta * J @ s[t].T
            term1 += C @ s[t].T
            term2 -= np.sum(np.log(np.exp(C) + np.exp(-C)))

            # term3 for including structural connectivity
            if type(sc) != type(None):
                diff_arr = np.power(J - np.dot(np.sign(J), sc), 2)
                total_diff_arr = (np.sum(diff_arr) - np.sum(diff_arr.diagonal())) / 2
                term3 -= 0.5 * lambda_ * total_diff_arr

        # print(f"C: {C.shape}, J: {J.shape}, s: {s.shape}, "  \
        #     + f"term1: {term1.shape}, term2: {term2.shape}")
        return -(term1+term2+term3)/self.n_timesteps

    def __gradient(self, J, s, beta, sc=None, lambda_=None):
        J = np.reshape(J, (self.n_rois, self.n_rois))
        grad = np.zeros((self.n_rois, self.n_rois))
        for t in range(self.n_timesteps):
            C = beta * J @ s[t].T
            grad += np.outer(s[t], s[t].T) - np.outer(np.tanh(C).T, s[t])

            # For structural connectivity inclusion
            if type(sc) != type(None):
                grad -= lambda_ * (J - np.dot(np.sign(J), sc))
            
        grad = grad * beta/self.n_timesteps
        return -grad.flatten()


    def gradient_descent(self, max_iterations,w_init,
                        obj_func,grad_func,extra_param = (),
                        learning_rate=0.05,momentum=0.8, threshold=0.001, disp=False):
        
        w = w_init
        bold_bin, beta, sc, lambda_ = extra_param
        w_history = [w]
        f_history = [obj_func(w, bold_bin, beta, sc, lambda_)]
        delta_w = np.zeros(w.shape)
        i = 0
        diff = 1.0e10
        
        while i<max_iterations and diff > threshold:
            grad = grad_func(w, bold_bin, beta, sc, lambda_)
            # print("from func", grad.shape)
            grad = np.reshape(grad, (self.n_rois, self.n_rois))
            # print(grad.shape)
            delta_w = -learning_rate*grad
            w = w+delta_w
            f_history.append(obj_func(w, bold_bin, beta, sc, lambda_))
            w_history.append(w)
            if i%10 == 0 and disp: 
                print(f"iteration: {i} loss: {f_history[-1]} grad: {np.sum(grad)}")
            i+=1
            diff = np.absolute(f_history[-1]-f_history[-2])
        w_max = w_history[f_history.index(min(f_history))]
        return w_max.flatten()

    def beta_optimization(self, bold, bold_bin, beta, sfc, sc=None, lambda_=None, id=None, n_folds = 5):
        print(f"Optimizing for beta = {beta}", flush=True)
        J = np.random.uniform(0, 1, size=(self.n_rois, self.n_rois))
        J = (J + J.T)/2 # making it symmetric
        np.fill_diagonal(J, 0)
        #fc = 1/self.n_timesteps * bold.T @ bold
        fc = sfc 
        # J_max = optimize.fmin_cg(loss, x0=J.flatten(), fprime=gradient, args=(bold, beta))
        J_max = self.gradient_descent(self.iterations, J, self.__loss, 
            self.__gradient, extra_param=(bold_bin, beta, sc, lambda_) , 
            learning_rate=self.alpha, threshold=0.005, disp=False)
        J_max = np.reshape(J_max, (self.n_rois, self.n_rois))
        sim = IsingSimulation(self.n_rois, beta, coupling_mat = True, J=J_max)
        for i in range(self.eq_steps):
            sim.step()
        corr = 0
        for i in range(n_folds):
            _, sim_fc = sim.getTimeseries(self.sim_timesteps)
            corr += np.corrcoef(fc, sim_fc[np.triu_indices(self.n_rois)].flatten())[0][1]
        
        if SAVE_EXTENDED:
            self.extended_save(id, J_max, beta, lambda_, corr/n_folds, n_folds)
        return J_max, corr/n_folds
    
    def beta_optimization_wrapper(self, bold, sfc, sc=None, lambda_=None, id=None):
        if type(lambda_) != type(None):
            print(f"Optimizing for lambda = {lambda_}")
        bold_bin = np.copy(bold)
        bold_bin[np.where(bold_bin >= 0)] = 1
        bold_bin[np.where(bold_bin < 0)] = -1
        results = Parallel(n_jobs=20)(
            delayed(self.beta_optimization)(bold, bold_bin, i, sfc, sc, lambda_, id=id) 
            for i in self.beta_range 
            )
        Js, corrs = np.array(results, dtype=object).T
        Js, corrs = list(Js), list(corrs)
        max_idx = np.argmax(corrs)
        beta_max, J_max = self.beta_range[max_idx], Js[max_idx]
        J_corr = np.corrcoef(sfc, J_max[np.triu_indices(self.n_rois)].flatten())[0][1]
        print('J corr', J_corr)
        return J_max, beta_max, corrs[max_idx]

    def lambda_optimization_wrapper(self, bold, sfc, sc, id=None):
        results = Parallel(n_jobs=20)(
            delayed(self.beta_optimization_wrapper)(bold, sfc, sc, i, id) 
            for i in self.lambda_range 
            )
        Js, betas, corrs = np.array(results, dtype=object).T
        max_idx = np.argmax(corrs)
        lambda_max = self.lambda_range[max_idx]
        J_max, beta_max, corr_max = Js[max_idx], betas[max_idx], corrs[max_idx]
        return J_max, beta_max, corr_max, lambda_max
    
    def optimize(self, bold, sfc, sc=None, id=None):
        opt_params = {}
        if type(sc) == type(None):
            print(f"Optimize called with no structural data")
            J_max, beta_max, corr_max = self.beta_optimization_wrapper(bold, sfc, id=id)
            opt_params['beta'] = beta_max
        else:
            print(f"Optimize called with structural data")
            J_max, beta_max, corr_max, lambda_max = \
                self.lambda_optimization_wrapper(bold, sfc, sc, id=id)
            opt_params['beta'] = beta_max
            opt_params['lambda'] = lambda_max
        return J_max, corr_max, opt_params

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
        if self.timeseries_result:
            return self.timeseries_result

        excluded_subjects = pd.DataFrame(columns=self.meta_data.columns)
        if not self.pre_parcellated:
            self.parcellate()

        timeseries = []
        IDs_subject = []
        age = []
        sex = []
        diagnosis = []
        atlas = self.atlas
        subject_ids = self.meta_data['SUB_ID']
        timeseries = []
        t_max = 0
        # For ABIDE
        for i_file in range(len(self.func)):
            f = self.func[i_file]
            # Uncomment for ABIDE
                # try:
                #     subject_id = int(f[f.find('5'):].split('_')[0])
                # except:
                #     print(f)
                #     subject_id = 'x'
                #     assert False, "remove above file from atlas folder"
            # Use for mica-mics
            subject_id = f.split("_")[0]

            # print(f"subject id: {subject_id}")
            this_pheno = self.meta_data[self.meta_data['SUB_ID'] == subject_id]
            this_timeseries = join(self.timeseries_dir, atlas, f)
            print(f"{i_file}: {this_timeseries}")
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
                IDs_subject.append(subject_id)
                diagnosis.append(this_pheno['DX_GROUP'].values[0])
                age.append(this_pheno['AGE_AT_SCAN'].values[0])
                sex.append(this_pheno['SEX'].values[0])
        # Padding to fix anomalous scans
        for i, t in enumerate(timeseries):
            timeseries[i] = pad_along_axis(t, t_max, 0)
        diagnosis = np.array(diagnosis) - 1
        timeseries = np.array(timeseries) 
        excluded_subjects.to_csv(join('../../data/excluded', atlas.replace('/', '-') +'.csv'))    
        self.n_timesteps = timeseries.shape[1]
        self.n_rois = timeseries.shape[2]
        print(f"timeseries shape: {timeseries.shape}, n_timesteps: {self.n_timesteps}, n_rois: {self.n_rois}")

        self.timeseries_result = (timeseries, IDs_subject, diagnosis, age, sex)
        return timeseries, IDs_subject, diagnosis, age, sex

    def SC(self):
        if self.SC_data:
            return self.SC_data

        _, ID, _, _, _ = self.get_timeseries()
        if (type(self.struct_conn_dir) == type(None)) or \
            (not os.path.exists(self.struct_conn_dir)):
            self.SC_data = [ None ] * len(ID)
            print("Structural data not found or not included")
            return self.SC_data

        sc_data = []
        for i in ID:
            fp = os.path.join(self.struct_conn_dir, i + "_sc.txt")
            w = np.loadtxt(fp)
            sc_data.append(w)
            # print(f"{i}: {fp}", flush=True)
        self.SC_data = sc_data

        return self.SC_data
    
    def sFC(self):
        if self.sFC_result:
            return self.sFC_result

        data, ID, diag, age, sex = self.get_timeseries()
        #correlation_measure = ConnectivityMeasure(kind='correlation', vectorize=True)
        #correlation_matrices = correlation_measure.fit_transform(data)
        corr = []
        for i in data:
            c = np.corrcoef(i.T)
            c = c[np.triu_indices(self.n_rois)].flatten()
            corr.append(c)
        corr = np.array(corr)

        self.sFC_result = (corr, ID, diag, age, sex)
        return corr, ID, diag, age, sex
    
    def ising_optimize_cg(self, bold, beta, J): 
        J_max = optimize.fmin_cg(self.__loss, x0=J.flatten(), \
            fprime=self.__gradient, args=(bold, beta), disp=False)
        return J_max
    
    def ising_optimize_gd(self, bold, beta, J): 
        J_max = self.gradient_descent(self.iterations, J, self.__loss, \
            self.__gradient, extra_param=(bold, beta) , \
            learning_rate=self.alpha, threshold=0.005, disp=False)
        J_max = J_max.reshape(self.n_rois, self.n_rois)
        #if not np.all(np.abs(J_max-J_max.T) < 1e-8):
            #print('not symmetric')

        J_max = J_max[np.triu_indices(J_max.shape[0], k = 1)].flatten()
        return J_max

    def ising_coupling(
                    self, 
                    iterations=500, 
                    alpha=2, 
                    beta_range=np.linspace(0.01, 0.105, 10), 
                    lambda_range=np.linspace(1e-7, 1e-5, 5), 
                    sim_timesteps = 500,
                    eq_timesteps=100
                    ):
        # setting parameters for GD
        self.eq_steps = eq_timesteps
        if sim_timesteps == -1: 
            self.sim_timesteps = self.n_timesteps
        else: self.sim_timesteps = sim_timesteps
        self.iterations = iterations
        self.alpha = alpha
        self.beta_range = [round(i, 2) for i in beta_range]
        self.lambda_range = lambda_range
        data, ID, diag, age, sex = self.get_timeseries()
        sfc, ID, diag1, age, sex = self.sFC()
        sc = self.SC()
        data_bin = np.copy(data)
        data_bin[np.where(data_bin >= 0)] = 1
        data_bin[np.where(data_bin < 0)] = -1
        J = np.random.uniform(0, 1, size=(self.n_rois, self.n_rois))
        J = (J + J.T)/2 # making it symmetric
        np.fill_diagonal(J, 0)
        reps = []
        betas = []
        diag = np.array(diag) - 1
        reps = np.zeros((len(data), self.n_rois, self.n_rois))
        betas = np.zeros(len(data))
        # if (type(sc) != type(None)) and (type(sc[0]) != type(None)):
        if not is_none_array(sc):
            lambdas = np.zeros(len(data))
        else:
            lambdas = [None] * len(data)
        corrs = np.zeros(len(data))
        print(f'Ising coupling: length of data = {len(data)}\n')
        print(f'sc: {len(sc)}')
        for idx, i in enumerate(data):
            print(f'{idx} - Subject: {ID[idx]}, shape: {i.shape}', flush=True)
            # J, b, c = self.beta_optimization_wrapper(i, sfc[idx], sc[idx])
            # reps[idx] = J
            # betas[idx] = b
            # corrs[idx] = c
            J, c, opt_params = self.optimize(i, sfc[idx], sc[idx], id=ID[idx])
            reps[idx] = J
            betas[idx] = opt_params['beta']
            corrs[idx] = c
            if opt_params.get('lambda', None):
                lambdas[idx] = opt_params['lambda']
                print(f"{idx}: subject = {ID[idx]}, best corr = {c}, beta = {betas[idx]}, lambda = {lambdas[idx]}")
            if SAVE_BY_SUBJECTS:
                self.save_subject(idx, ID[idx], J, sfc[idx], sc[idx], \
                    diag[idx], age[idx], sex[idx], c, betas[idx], lambdas[idx])
        return reps, betas, corrs, ID, diag, age, sex, lambdas

    def compute_corrs(self, J, sfc, sc=None):
        sfc_ut = sfc
        J_ut = J[np.triu_indices(self.n_rois)]
        sc_ut = sc[np.triu_indices(self.n_rois)]
        J_sfc_corr = np.corrcoef(J_ut, sfc_ut)[0][1]

        sfc_sc_corr, J_sc_corr = None, None
        if type(sc) != type(None):
            sfc_sc_corr = np.corrcoef(sc_ut, sfc_ut)[0][1]
            J_sc_corr = np.corrcoef(sc_ut, J_ut)[0][1]

        return J_sfc_corr, J_sc_corr, sfc_sc_corr

    def extended_save(self, id, J, beta, lambda_, corr, n_folds):
        if not os.path.exists(EXTENDED_SAVE_FP):
            data = []
            with open(EXTENDED_SAVE_FP, 'wb') as f:
                pkl.dump(data, f)
        new_data = {
            'id' : id,
            'J' : J,
            'beta' : beta,
            'lambda' : lambda_,
            'corr' : corr, 
            'n_folds' : n_folds,
        }
        with open(EXTENDED_SAVE_FP, 'rb') as f:
            data = pkl.load(f)
        data.append(new_data)
        with open(EXTENDED_SAVE_FP, 'wb') as f:
            pkl.dump(data, f)
        return

    def save_subject(self, idx, id, J, sfc, sc, diag, age, sex, c, beta, lambda_):
        J_sfc_corr, J_sc_corr, sfc_sc_corr = self.compute_corrs(J, sfc, sc)
        results = {
            'idx' : idx, 'id': id,
            'J' : J, 'sfc': sfc, 'sc': sc,
            'diag' : diag, 'age' : age, 'sex' : sex,
            'corr': c, 'beta' : beta, 'lambda': lambda_,
            'J_sfc_corr' : J_sfc_corr,
            'J_sc_corr' : J_sc_corr,
            'sfc_sc_corr' : sfc_sc_corr
        }
        fp = os.path.join(SUBJECTS_SAVE_DIR, f"{id}_results.pkl")
        with open(fp, 'wb') as f:
            pkl.dump(results, f)
        return
        
    pass

if __name__ == '__main__':
    atlas = ''
    sites = [ 'main' ]
    n_rois = 164
    for site in sites:
        dataset = Abide(sites=sites, atlas=atlas)
        betas = []
        diag = []
        corr = []
        ising = []
        J_corr = []
        sfc, sub, diag, age, sex = dataset.sFC()
        ising, betas, corr, sub1, diag1, age1, sex1, lambdas = dataset.ising_coupling()
        sc = dataset.SC()

        np.save(os.path.join(SAVE_DIR, f'diag_{site}.npy'), diag)
        np.save(os.path.join(SAVE_DIR, f'sub_{site}.npy'), sub)
        np.save(os.path.join(SAVE_DIR, f'sc_{site}.npy'), sc)
        np.save(os.path.join(SAVE_DIR, f'sfc_{site}.npy'), sfc)
        np.save(os.path.join(SAVE_DIR, f'ising_{site}.npy'), ising)
        np.save(os.path.join(SAVE_DIR, f'betas_{site}.npy'), betas)
        np.save(os.path.join(SAVE_DIR, f'corr_{site}.npy'), corr)
        np.save(os.path.join(SAVE_DIR, f'lambdas_{site}.npy'), lambdas)

        print('ising shape:', ising[0][np.triu_indices(n_rois)].flatten().shape)
        print('sfc shape:', sfc[0].flatten().shape)
        print('fc and j corr:')
        for i in range(ising.shape[0]):
            k = np.corrcoef(sfc[i].flatten(), ising[i][np.triu_indices(n_rois)].flatten())[0][1]
            J_corr.append(k)
            print(k)
        J_corr = np.array(J_corr)
        assert np.array_equal(sub, sub1)
        np.save(os.path.join(SAVE_DIR, f'J_fc_corr_{site}.npy'), J_corr)

        # if (type(sc) != type(None)) and (type(sc[0]) != type(None)):
        if not is_none_array(sc):
            J_sc_corr = []
            for i in range(ising.shape[0]):
                k = np.corrcoef(sc[i].flatten(), ising[i][np.triu_indices(n_rois)].flatten())[0][1]
                J_sc_corr.append(k)
                J_sc_corr = np.array(J_sc_corr)
            np.save(os.path.join(SAVE_DIR, f'J_sc_corr_{site}.npy'), J_sc_corr)

