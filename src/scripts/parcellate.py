import os
from nilearn.input_data import NiftiLabelsMasker
from os.path import join
from joblib import Parallel, delayed
from nilearn import datasets
import numpy as np

func_files_dir = '/scratch/shivansh.seth/func_preproc'
func_files = os.listdir(func_files_dir)
cc200 = datasets.fetch_atlas_craddock_2012()
aal = datasets.fetch_atlas_aal()
aal = aal.maps
cc200 = cc200.scorr_2level
timeseries_dir = '/home/shivansh.seth/parc'
#os.mkdir(timeseries_dir)
multiscale = datasets.fetch_atlas_basc_multiscale_2015()
atlas_122 = multiscale.scale122

def parcellate(i, atlas, masker):
    sub_id = int(i[i.find('5'):].split('_')[0])
    print(sub_id)
    file_name = join(func_files_dir, i)
    k = masker.fit_transform(file_name)
    np.savetxt(join(timeseries_dir, atlas, str(sub_id) + '_timeseries.txt'), k)
    return k

for atlas_file in [atlas_122]:
    os.mkdir(join(timeseries_dir, 'basc_122'))
    masker = NiftiLabelsMasker(labels_img=atlas_file,
                                memory='nilearn_cache')
    print('getting parcellated time series...')
    #res = parcellate(func_files[0], 'basc_122', masker)
    res = Parallel(n_jobs=20)(delayed(parcellate)(i, 'basc_122', masker) for i in func_files)
    print(res[0].shape)
    print('atlas done')
