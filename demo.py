from sklarge import GridSearchCV, run_local, evaluation, run_condor

from sklearn.linear_model import Ridge
from sklearn import datasets
import numpy as np

boston = datasets.load_boston()
X, y = boston.data, boston.target

# add some fake label
y = np.array([y,y*2,np.zeros_like(y)]).T

# split data in folds
X = [X[:400],X[400:]]
y = [y[:400],y[400:]]

# define experiments
idx = [
        [ [0] , [1] ],
        [ [1] , [0] ],
      ]

clf = Ridge()
param_grid = {'alpha':[0.01, 0.1, 1., 10., 100.]}


GS = GridSearchCV(
        estimator = clf,
        param_grid = param_grid,
        )

GS._create_job_files(X, y, idx, out_path='tmpg', mode = 'w')
run_condor('tmpg',1,graphic_only=1)

GS._create_job_files(X, y, idx, out_path='tmpc', mode = 'w')
run_condor('tmpc',1,graphic_only=0)

GS._create_job_files(X, y, idx, out_path='tmp2', mode = 'w')
run_local('tmp2',1)

evaluation('tmpc')
evaluation('tmpg')
evaluation('tmp2')
