from sklarge.model_selection import GridSearchCV, LabelKFold, Eval
from sklearn.linear_model import Ridge
import h5py


f = h5py.File('tests/data/test.h5')
cv = LabelKFold(3)


clf = Ridge()
param_grid = {'alpha':[0.01, 0.1, 1., 10., 100.]}

GS = GridSearchCV(
        estimator = clf,
        param_grid = param_grid,
        cv = LabelKFold(3),
        n_jobs = -1,
        out_path = '.tmp'
        )

GS.fit(
        X = f['X'][::],
        y = f['y'][::],
        submit='local',
        )
Eval('.tmp')
