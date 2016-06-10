import numpy as np
from sklearn.multioutput import MultiOutputClassifier as MC
from sklearn.multioutput import MultiOutputRegressor as MR


from sklearn.linear_model import Ridge as _Ridge
class MVR(_Ridge):
    param_grid = {'alpha':10.**np.arange(-5,6)}
    def fit(self,X,y,mask=False):
        if np.any(mask):
            X = X[mask.tolist()]
            y = y[mask.tolist()]
        _Ridge.fit(self,X,y)
        return self
    def predict(self,X,mask=False):
        if np.any(mask):
            X = X[mask.tolist()]
        y_hat = _Ridge.predict(self,X)
        return y_hat

from sklearn.linear_model import MultiTaskLasso as _MultiTaskLasso
class MTL(_MultiTaskLasso):
    param_grid = {'alpha':10.**np.arange(-5,6)}
    def fit(self,X,y,mask=False):
        if np.any(mask):
            X = X[mask.tolist()]
            y = y[mask.tolist()]
        _MultiTaskLasso.fit(self,X,y)
        return self
    def predict(self,X,mask=False):
        if np.any(mask):
            X = X[mask.tolist()]
        y_hat = _MultiTaskLasso.predict(self,X)
        return y_hat

from sklearn.svm import SVR as libSVR
class SVR():
    estimator = MR(libSVR(max_iter=1000))
    param_grid = [
            {
                'estimator__C': 10.**np.arange(-3,3),
                'estimator__kernel': ['linear'],
                'estimator__class_weight':['balanced',None],
            },
            {
                'estimator__C': 10.**np.arange(-3,3),
                'estimator__kernel': ['rbf','sigmoid'],
                'estimator__gamma': 10.**np.arange(-3,3),
                'estimator__class_weight':['balanced',None],
            }]
    def fit(self,X,y,mask=False):
        if np.any(mask):
            X = X[mask.tolist()]
            y = y[mask.tolist()]
        self.estimator.fit(X,y)
        return self
    def predict(self,X,mask=False):
        if np.any(mask):
            X = X[mask.tolist()]
        y_hat = self.estimator.predict(X)
        return y_hat

from sklearn.svm import SVC as libSVC
class SVC():
    estimator = MC(libSVC(max_iter=1000))
    param_grid = [
            {
                'estimator__C': 10.**np.arange(-3,3),
                'estimator__kernel': ['linear'],
                'estimator__class_weight':['balanced',None],
            },
            {
                'estimator__C': 10.**np.arange(-3,3),
                'estimator__kernel': ['rbf','sigmoid'],
                'estimator__gamma': 10.**np.arange(-3,3),
                'estimator__class_weight':['balanced',None],
                'estimator__max_iter':[1000]
            }]
    def fit(self,X,y,mask=False):
        if np.any(mask):
            X = X[mask.tolist()]
            y = y[mask.tolist()]
        self.estimator.fit(X,y)
        return self
    def predict(self,X,mask=False):
        if np.any(mask):
            X = X[mask.tolist()]
        y_hat = self.estimator.predict(X)
        return y_hat
