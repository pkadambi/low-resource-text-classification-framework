from sklearnex import patch_sklearn
patch_sklearn()
import tqdm
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
from scipy.spatial import distance
import torch
import torch.nn.functional as F
from sklearn.calibration import calibration_curve
import pickle as pkl
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from calibration_metrics import *


dct = pkl.load(open('../data/fashionmnist_with_vae.pkl', 'rb'))
_xtrain = np.concatenate([item[2] for item in dct['train']])
_xtrain_std = np.concatenate([item[3] for item in dct['train']])
_ytrain = np.array([item[1] for item in dct['train']])
_ytrposterior = dct['train']

_xtest = np.concatenate([item[2] for item in dct['test']])
_xteststd = np.concatenate([item[3] for item in dct['test']])
_ytest = np.array([item[1] for item in dct['test']])


def estimate_posterior(xdata, ytrain, kvalue=10):
    from sklearn.neighbors import KNeighborsClassifier
    print(f'Estimating posterior, Using k={kvalue}')
    neigh = KNeighborsClassifier(n_neighbors=kvalue)
    neigh.fit(xdata, ytrain)
    knnposteriors = []
    for ii in tqdm.tqdm(range(xdata.shape[0])):
        datapoint = xdata[ii, :].reshape(1, -1)
        neigh_inds = neigh.kneighbors(datapoint, return_distance=False)
        _yneighb = torch.tensor(ytrain[neigh_inds]).to(torch.int64)[0]
        estimated_label = F.one_hot(_yneighb, num_classes=10).to(torch.float).numpy().mean(axis=0)
        knnposteriors.append(estimated_label)
    return knnposteriors

dim = 3
xtrain = np.concatenate([np.random.randn(10000,dim), np.random.randn(10000, dim) + 2])
xtest = np.concatenate([np.random.randn(5000,dim), np.random.randn(5000, dim) + 2])
xcal = np.concatenate([np.random.randn(5000, dim), np.random.randn(5000, dim) + 2])

ytrain = np.concatenate([np.zeros([10000, 1]), np.ones([10000, 1])]).ravel()
ytest = np.concatenate([np.zeros([5000,1]), np.ones([5000, 1])]).ravel()
ycal = np.concatenate([np.zeros([5000, 1]), np.ones([5000, 1])]).ravel()

clfrf = RandomForestClassifier(max_depth=2, random_state=0)
clfrf.fit(xtrain, ytrain)

ycalpredrf = clfrf.predict(xtest)
print(sum(ycalpredrf==ycal)/len(ycal))
ycalpostrf = clfrf.predict_proba(xtest)


clflr = LogisticRegression()
clflr.fit(xtrain, ytrain)

ycalpredlr = clflr.predict(xtest)
print(sum(ycalpredlr==ycal)/len(ycal))
ycalpostlr = clflr.predict_proba(xtest)

clfsv = SVC(probability=True)
clfsv.fit(xtrain, ytrain)
ycalpredsv = clfsv.predict(xtest)
ycalpostsv = clfsv.predict_proba(xtest)
print(sum(ycalpredsv==ycal)/len(ycal))




print('Random Forest')
print(return_all_calibration_metrics(ycal, ycalpostrf, n_bins=10, nclasses=2, binning_method='quantile'))

print('SVM')
print(return_all_calibration_metrics(ycal, ycalpostsv, n_bins=10, nclasses=2, binning_method='quantile'))

print('Logistic Regression')
print(return_all_calibration_metrics(ycal, ycalpostlr, n_bins=10, nclasses=2, binning_method='quantile'))













