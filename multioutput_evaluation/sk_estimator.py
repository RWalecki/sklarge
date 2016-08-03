import numpy as np
from sklearn.multioutput import MultiOutputClassifier as MC
from sklearn.multioutput import MultiOutputRegressor as MR

def preprocessing(X,y=None):

    # load data into memory
    X = X[::]

    # concatonate all features if X has more than 2 dimensions
    X = X.reshape(X.shape[0],-1)

    # todo: preprocess the data and remove this
    X[np.isnan(X)]=0

    if y!=None:
        y = y[::]
        y[y==9]=0
        return X,y
    return X


from sklearn.linear_model import Ridge as _Ridge
class MVR():
    param_grid = {'alpha':10.**np.arange(-5,6)}
    estimator = _Ridge()
    preprocessing = preprocessing
    def fit(self,X,y,mask=False):
        if np.any(mask):
            X = X[mask,:]
            y = y[mask,:]
        X,y = preprocessing(X,y)
        self.estimator.fit(X,y)
        return self
    def predict(self,X,mask=False):
        if np.any(mask):
            X = X[mask,:]
        X = preprocessing(X)
        return self.estimator.predict(X)
    def set_params(self,**args):
        self.estimator.set_params(**args)

from sklearn.linear_model import MultiTaskLasso as _MultiTaskLasso
class MTL():
    param_grid = {'alpha':10.**np.arange(-6,10)}
    estimator = _MultiTaskLasso()
    def fit(self,X,y,mask=False):
        if np.any(mask):
            X = X[mask,:]
            y = y[mask,:]
        X,y = preprocessing(X,y)
        self.estimator.fit(X,y)
        return self
    def predict(self,X,mask=False):
        if np.any(mask):
            X = X[mask,:]
        X = preprocessing(X)
        y_hat = self.estimator.predict(X)
        return y_hat
    def set_params(self,**args):
        self.estimator.set_params(**args)

from sklearn.svm import SVR as libSVR
class SVR():
    estimator = MR(libSVR(max_iter=2000))
    param_grid = [{
            'estimator__C': 10.**np.arange(-3,3),
            'estimator__kernel': ['linear'],
            },
            {
            'estimator__C': 10.**np.arange(-3,3),
            'estimator__kernel': ['rbf','sigmoid'],
            'estimator__gamma': 10.**np.arange(-3,3),
            }]
    def fit(self,X,y,mask=False):
        if np.any(mask):
            X = X[mask,:]
            y = y[mask,:]
        X,y = preprocessing(X,y)
        self.estimator.fit(X,y)
        return self
    def predict(self,X,mask=False):
        if np.any(mask):
            X = X[mask,:]
        X = preprocessing(X)
        y_hat = self.estimator.predict(X)
        return y_hat
    def set_params(self,**args):
        self.estimator.set_params(**args)

from sklearn.svm import SVC as libSVC
class SVC():
    estimator = MC(libSVC(max_iter=2000))
    param_grid = [{
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
        X,y = preprocessing(X,y)
        self.estimator.fit(X,y)
        return self
    def predict(self,X,mask=False):
        if np.any(mask):
            X = X[mask.tolist()]
        X = preprocessing(X)
        y_hat = self.estimator.predict(X)
        return y_hat
    def set_params(self,**args):
        self.estimator.set_params(**args)
