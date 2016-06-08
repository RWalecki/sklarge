import numpy as np
from sklearn.multioutput import MultiOutputClassifier as MC
from sklearn.multioutput import MultiOutputRegressor as MR


from sklearn.linear_model import Ridge as _Ridge
class MVR():
    estimator = _Ridge()
    parameter = {'alpha':10.**np.arange(-5,6)}

from sklearn.neighbors import KNeighborsClassifier as _KNeighborsClassifier
class KNN(_KNeighborsClassifier):
    estimator = _KNeighborsClassifier()
    parameter = {'n_neighbors': 2**np.arange(1,8),'weights':['uniform','distance']}

from sklearn.linear_model import MultiTaskLasso as _MultiTaskLasso
class MTL():
    estimator = _MultiTaskLasso()
    parameter = {'alpha':10.**np.arange(-5,6)}

from sklearn.ensemble import RandomForestClassifier as _RandomForestClassifier
class RF(_RandomForestClassifier):
    estimator = _RandomForestClassifier()
    parameter = {'n_estimators': 2**np.arange(2,6)}

from sklearn.linear_model import SGDClassifier as _SGDClassifier
class SVC():
    estimator = MC(_SGDClassifier(n_iter=100))
    parameter = {'estimator__alpha': 10.**np.arange(-5,6)}

from sklearn.linear_model import SGDRegressor as _SGDRegressor
class SVR():
    estimator = MR(_SGDRegressor(n_iter=100))
    parameter = {'estimator__alpha': 10.**np.arange(-5,6)}

from sklearn.svm import SVC as _SVC
class libSVC():
    estimator = MC(_SVC(max_iter=1000))
    parameter = [
            {
                'estimator__C': 10.**np.arange(-3,3),
                'estimator__kernel': ['linear'],
                'estimator__class_weight':['balanced',None]
            },
            {
                'estimator__C': 10.**np.arange(-3,3),
                'estimator__kernel': ['rbf','sigmoid'],
                'estimator__gamma': 10.**np.arange(-3,3),
                'estimator__class_weight':['balanced',None]
            }
            ]

from sklearn.svm import SVR as _SVR
class libSVR():
    estimator = MC(_SVR(max_iter=1000))
    parameter = [
            {
                'estimator__C': 10.**np.arange(-3,3),
                'estimator__kernel': ['linear'],
            },
            {
                'estimator__C': 10.**np.arange(-3,3),
                'estimator__kernel': ['rbf','sigmoid'],
                'estimator__gamma': 10.**np.arange(-3,3),
            }
            ]
