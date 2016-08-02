from sklearn.multioutput import MultiOutputClassifier as MC
import numpy as np
from estimator import ordinal_regression
from .sk_estimator import preprocessing 


class SOR():
    estimator=MC(ordinal_regression(verbose=1))
    def __init__(
            self,
            verbose = 1
            ):
        self.verbose = verbose

    param_grid = {
            'estimator__learning_rate': [0.0001,0.001,0.01,0.1,1.],
            'estimator__training_epochs': [10000],
            # 'estimator__optimizer': ['SGD','Adam','Adagrad'],
            'estimator__batch_size': [-1,1000],
            }
    preprocessing = preprocessing
    def fit(self,X,y,mask=False):
        if np.any(mask):
            X = X[mask,:]
            y = y[mask,:]
        X,y = preprocessing(X,y)
        n_classes = len(np.unique(y))

        self.estimator.fit(np.float32(X),np.float32(y))
        return self
    def predict(self,X,mask=False):
        if np.any(mask):
            X = X[mask,:]
        X = preprocessing(X)
        return self.estimator.predict(X)
    def set_params(self,**args):
        self.estimator.set_params(**args)
