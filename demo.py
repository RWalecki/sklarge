from sklarge import GridSearchCV, run_local, evaluation, run_condor
from sklearn.model_selection import KFold

from sklearn.linear_model import Ridge
from sklearn import datasets
import numpy as np

boston = datasets.load_boston()
X, y = boston.data, boston.target

# add some fake label
y = np.array([y,y*2,np.zeros_like(y)]).T

# split data in folds
X_set = []
Y_set = []
for i in range(5):
    X_set.append(X[i*100:(i+1)*100])
    Y_set.append(y[i*100:(i+1)*100])

# define folds 
cv = KFold(3)
idx = [f for f in cv.split(range(len(X_set)))]

clf = Ridge()
param_grid = {
        'alpha':[0.01, 0.1, 1., 10.],
        'normalize': [True, False]
        }

GS = GridSearchCV(
        estimator = clf,
        param_grid = param_grid,
        )

GS._create_job_files(X_set, Y_set, idx, out_path='tmp_jobs', mode = 'w')
run_local('tmp_jobs',-1)

out = evaluation('tmp_jobs', best_joint=True, verbose=2,condition={'normalize':False})
# out = evaluation('tmp_jobs', best_joint=True, verbose=2)
print(out['table'].shape)
print(out['best_params'])

import MyScripts
MyScripts.latex.numpy_to_latex(
        out['table'],
        columns = out['columns'],
        index = out['index'],
        verbose=10,
        bold=['max','h'],
        path='tmp.tex'
        )
