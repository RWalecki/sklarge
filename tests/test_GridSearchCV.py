import os
import h5py 
import numpy as np

from sklearn.model_selection import GridSearchCV as GridSearchCV_old
from sklearn.model_selection import LabelKFold
from sklearn.linear_model import Ridge

from sklarge.model_selection import GridSearchCV, Eval
from sklarge.metrics import mse, pcc

pwd = os.path.dirname(os.path.abspath(__file__))


f = h5py.File(pwd+'/data/test.h5')
clf = Ridge()
param_grid = {'alpha':10.**np.arange(-2,3)}

class testcase:

    def test_sk_estimator(self):
        clf.fit(f['X'],f['y'])
        y_hat = clf.predict(f['X'])
        assert y_hat.shape == f['y'].shape

    def test_grid_search_basic(self):
        GS = GridSearchCV(
                estimator = clf,
                param_grid = param_grid,
                cv = LabelKFold(2),
                n_jobs = -1,
                )
        GS.fit(
                X = f['X'],
                y = f['y'][:,5],
                tmp = pwd+'/tmp/',
                )
        Eval(pwd+'/tmp/'+clf.__class__.__name__)

    def test_grid_search_basic_multi_output(self):
        GS = GridSearchCV(
                estimator = clf,
                param_grid = param_grid,
                cv = LabelKFold(2),
                n_jobs = -1,
                )
        GS.fit(
                X = f['X'],
                y = f['y'],
                tmp = pwd+'/tmp/',
                )
        Eval(pwd+'/tmp/'+clf.__class__.__name__)

    def test_grid_search_different_scoring(self):
        GS = GridSearchCV(
                estimator = clf,
                param_grid = param_grid,
                cv = LabelKFold(2),
                scoring=[pcc,mse],
                n_jobs = -1,
                verbose = 2,
                )
        GS.fit(
                X = f['X'],
                y = f['y'],
                tmp = pwd+'/tmp/',
                )
        Eval(pwd+'/tmp/'+clf.__class__.__name__)

    def test_grid_search_different_numpy_input(self):
        GS = GridSearchCV(
                estimator = clf,
                param_grid = param_grid,
                cv = LabelKFold(2),
                scoring=[pcc,mse],
                n_jobs = -1,
                verbose = 2,
                )
        GS.fit(
                X = f['X'][::],
                y = f['y'][::],
                tmp = pwd+'/tmp/',
                )
        Eval(pwd+'/tmp/'+clf.__class__.__name__)
        GS.fit(
                X = f['X'],
                y = f['y'][::],
                tmp = pwd+'/tmp/',
                )
        Eval(pwd+'/tmp/'+clf.__class__.__name__)

    def test_grid_search_labels(self):
        GS = GridSearchCV(
                estimator = clf,
                param_grid = param_grid,
                cv = LabelKFold(2),
                scoring=[pcc,mse],
                n_jobs = -1,
                verbose = 2,
                )
        GS.fit(
                X = f['X'],
                y = f['y'],
                labels = np.arange(f['X'].shape[0]),
                tmp = pwd+'/tmp/',
                )
        Eval(pwd+'/tmp/'+clf.__class__.__name__)

    def test_compare_with_sk_gridseach(self):
        GS_old = GridSearchCV_old(
                estimator = clf,
                param_grid = {'alpha':10.**np.arange(-5,5)},
                scoring = mse,
                cv = LabelKFold(2),
                )

        GS_new = GridSearchCV(
                estimator = clf,
                param_grid = {'alpha':10.**np.arange(-5,5)},
                scoring = mse,
                cv = LabelKFold(2),
                )

        GS_old.fit(f['X'][::],f['y'][::,5],f['S'][::])
        GS_new.fit(f['X'][::],f['y'][::,5],f['S'][::])

        assert GS_old.best_score_-GS_new.get_best_score() < 1e-4
        assert GS_old.best_params_['alpha']==GS_new.get_best_param()['alpha']



if __name__ == "__main__":
    import nose
    nose.run(defaultTest=__file__, env={'NOSE_NOCAPTURE' : 1})
