from sklearn.model_selection import LabelKFold
from sklearn.linear_model import Ridge
from sklarge.model_selection import GridSearchCV, Eval
import h5py
import numpy as np


f = h5py.File('tests/data/test.h5')

clf = Ridge()
param_grid = {'alpha':10.**np.arange(-2,3)}

GS = GridSearchCV(
        estimator = clf,
        param_grid = param_grid,
        cv = LabelKFold(2),
        n_jobs = -1,
        out_path = '.tmp'
        )

GS.fit(
        X = f['X'][::],
        y = f['y'][::],
        submit='local',
        )
Eval('.tmp')
