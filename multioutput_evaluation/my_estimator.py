from sklearn.multioutput import MultiOutputClassifier as MC
import numpy as np
from estimator import ordinal_regression
from sk_estimator import preprocessing 


class SOR():
    def __init__(
            self,
            C = 0,
            training_epochs = 10000, 
            learning_rate = 0.001, 
            batch_size = -1, 
            verbose = 1
            ):
        self.C = C
        self.training_epochs = training_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.verbose = verbose

    param_grid = {
            'estimator__learning_rate': [0.001],
            # 'estimator__optimizer': ['SGD','Adam','Adagrad'],
            'estimator__batch_size': [-1],
            }
    preprocessing = preprocessing
    def fit(self,X,y,mask=False):
        if np.any(mask):
            X = X[mask,:]
            y = y[mask,:]
        X,y = preprocessing(X,y)
        n_classes = len(np.unique(y))
        self.estimator=MC(
                ordinal_regression(verbose=self.verbose, training_epochs=self.training_epochs) 
                )

        self.estimator.fit(np.float32(X),np.float32(y))
        return self
    def predict(self,X,mask=False):
        if np.any(mask):
            X = X[mask,:]
        X = preprocessing(X)
        return self.estimator.predict(X)
    def set_params(self,**args):
        pass
        # self.estimator.set_params(**args)
