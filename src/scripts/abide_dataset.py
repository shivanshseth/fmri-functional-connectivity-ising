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

    def parcellate(self):
        res = []
        atlas = self.atlas
        os.mkdir(join(self.timeseries_dir, atlas_filename))
        masker = NiftiLabelsMasker(labels_img=atlas_filename,
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
                excluded_subjects = excluded_subjects.append(this_pheno)
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
        correlation_measure = ConnectivityMeasure(kind='correlation', vectorize=True, discard_diagonal= True)
        correlation_matrices = correlation_measure.fit_transform(data)
        return correlation_matrices, ID, diag, age, sex
    
    
    def ising_coupling(self):
        data, ID, diag, age, sex = self.get_timeseries()
        data[np.where(data >= 0)] = 1
        data[np.where(data < 0)] = -1
        J = np.random.uniform(0, 1, size=(self.n_rois, self.n_rois))
        J = (J + J.T)/2 # making it symmetric
        np.fill_diagonal(J, 0)
        beta = 0.1
        reps = []
        diag = np.array(diag) - 1
        # idx = np.where(diag > 0)[0][:2]
        # data = np.vstack((data[:2], data[idx]))
        # diag = np.concatenate((diag[:2], diag[idx]))
        for i in data:
            J_max = optimize.fmin_cg(self.__loss, x0=J.flatten(), fprime=self.__gradient, args=(i, beta), disp=False)
            reps.append(J_max)
        reps = np.array(reps)
        return reps, ID, diag, age, sex
        

if __name__ == '__main__':
    dataset = Abide(sites='NYU')
    dataset.ising_coupling('AAL', 'AAL')