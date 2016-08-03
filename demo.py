from sklearn.model_selection import GridSearchCV as GridSearchCV_old
from sklearn.model_selection import LabelKFold
from sklearn.linear_model import Ridge

from sklarge.model_selection import GridSearchCV, Eval
from sklarge.metrics import mse, pcc
import os
import h5py
import numpy as np

pwd = os.path.dirname(os.path.abspath(__file__))


f = h5py.File(pwd+'/projects/sklarge/tests/data/test.h5')
print(f['X'])
print(f['y'])

clf = Ridge()
param_grid = {'alpha':10.**np.arange(-2,3)}

GS = GridSearchCV(
        estimator = clf,
        param_grid = param_grid,
        cv = LabelKFold(2),
        n_jobs = -1,
        )
GS.fit(
        X = f['X'][::],
        y = f['y'][::],
        tmp = pwd+'/tmp3/',
        submit='local',
        )
Eval(pwd+'/tmp3/'+clf.__class__.__name__)
