import numpy as np
import sys
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
from nitime.timeseries import TimeSeries
from nitime.analysis import SpectralAnalyzer, FilterAnalyzer, NormalizationAnalyzer
import matplotlib.pyplot as plt
from scipy import stats, signal
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn import svm
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from abide_dataset import Abide
from nilearn.connectome import ConnectivityMeasure
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

def train_test(
        scale, 
        rep, 
        rep_data, 
        aut_labels,
        df,
        sex_labels=None,
        age_labels=None,
        train_acc=False,
    ):
    # clf = svm.SVC(kernel='linear')
    # fs = SelectPercentile(f_classif, percentile=10)
    # rep_data = fs.fit_transform(rep_data, aut_labels)
    if train_acc: 
        X_train, X_test, y_train, y_test = train_test_split(rep_data, aut_labels, test_size=0.4, random_state=10)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_train)
        accuracy = accuracy_score(y_pred, y_train)
        print("Training Accuracy of Autism prediction: ", accuracy*100, "%")
    clf = Pipeline([
        ('feature_selection', SelectPercentile(percentile=10)),
        ('classification', svm.SVC(kernel='linear'))
    ])
    scoring = {
        'acc': 'accuracy',
        'f1': 'f1',
        'prec': 'precision',
        'rec': 'recall'
    }
    scores = cross_validate(clf, rep_data, aut_labels, scoring=scoring, cv = 10)  
#     print(scores)
#     print(accuracy)
    print("Accuracy of Autism prediction: ", scores['test_acc'].mean()*100, "%")
    df.loc[len(df)] = [
        scale, rep, 
        scores['test_acc'].mean()*100, 
        scores['test_acc'].std()*100, 
        scores['test_f1'].mean(), 
        scores['test_f1'].std(), 
        scores['test_prec'].mean(), 
        scores['test_prec'].std(), 
        scores['test_rec'].mean(),
        scores['test_rec'].std()
    ]
    return df
df = pd.DataFrame(columns = ['Atlas', 'Representation',  'Accuracy', 'acc_std', 'F1', 'f1_std', 'Precision', 'prec_std', 'Recall', 'rec_stds'])
dataset = Abide(sites='all', scale='AAL', atlas='AAL')
print("starting preprocessing")
#cgd, ID, diag, age, sex = dataset.ising_coupling(method='CG')
#gd, ID, diag, age, sex = dataset.ising_coupling(method='GD')
sfc, ID, diag, age, sex = dataset.sFC()
gd = np.load('../../data/gd_all.npy')
diag = np.load('../../data/diag_all.npy')
print("preprocessing finished")
print("starting training")
clf = DummyClassifier(strategy="most_frequent")
clf.fit(sfc, diag)
y_pred = clf.predict(sfc)
acc = accuracy_score(diag, y_pred)
print(f"most freq acc: {acc}")
# res_df = train_test('AAL', 'ising_cgd', cgd, diag, df, age, sex, True)
res_df = train_test('AAL', 'sFC', sfc, diag, df, age, sex, True)
res_df = train_test('AAL', 'ising_gd', gd, diag, df, age, sex, True)
res_df.to_csv('../results/classification_feature_selection.csv')
print(res_df)
