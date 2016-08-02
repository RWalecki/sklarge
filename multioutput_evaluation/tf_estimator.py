from sklearn.multioutput import MultiOutputClassifier as MC
from sklearn.multioutput import MultiOutputRegressor as MR
from tensorflow.contrib.learn.python import learn
import numpy as np
from .sk_estimator import preprocessing 

class DNN_C():
    def __init__(self, max_iter=1000, verbose=0):
        self.max_iter=max_iter
        self.verbose=verbose

    param_grid = {
            'estimator__hidden_units': [[10,20,10],[5,10,5],[100,10],[10,10,10,10]],
            'estimator__learning_rate': [0.1, 0.25, 0.5, 1., 2. ],
            'estimator__optimizer': ['SGD','Adam','Adagrad'],
            'estimator__batch_size': [16,32,64],
            }
    def fit(self,X,y,mask=False):
        if np.any(mask):
            X = X[mask,:]
            y = y[mask,:]
        X,y = preprocessing(X,y)
        n_classes = len(np.unique(y))
        self.estimator=MC(learn.TensorFlowDNNClassifier(
            hidden_units=[10, 20, 10], 
            n_classes=n_classes, 
            steps=self.max_iter,
            verbose=self.verbose)
            )

        self.estimator.fit(np.float32(X),np.int32(y))
        return self
    def predict(self,X,mask=False):
        if np.any(mask):
            X = X[mask,:]
        X = preprocessing(X)
        return self.estimator.predict(X.astype(np.int32))

    def set_params(self,**args):
        self.estimator.set_params(**args)

class DNN_R():
    def __init__(self, max_iter=1000, verbose=0):
        self.max_iter=max_iter
        self.verbose=verbose

    parameter = {
            'estimator__hidden_units': [[10,20,10],[5,10,5],[100,10],[10,10,10,10]],
            'estimator__learning_rate': [0.1, 0.25, 0.5, 1., 2. ],
            'estimator__optimizer': ['SGD','Adam','Adagrad'],
            'estimator__batch_size': [16,32,64],
            }
    def fit(self,X,y,mask=False):
        n_classes = len(np.unique(y))

        self.estimator=MR(learn.TensorFlowDNNRegressor(
            hidden_units=[10, 20, 10], 
            steps=self.max_iter,
            verbose=self.verbose)
            )
        if np.any(mask):
            X = X[mask.tolist()]
            y = y[mask.tolist()]
        self.estimator.fit(np.float16(X),y)
        return self
    def predict(self,X,mask=False):
        if np.any(mask):
            X = X[mask.tolist()]
        y_hat = self.estimator.predict(np.float16(X))
        return y_hat
