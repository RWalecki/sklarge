from sklearn.multioutput import MultiOutputClassifier as MC
from tensorflow.contrib.learn.python import learn
import numpy as np
class DNNC():
    estimator=MC(learn.TensorFlowDNNClassifier(hidden_units=[10, 20, 10], n_classes=6, steps=1000,verbose=0))
    parameter = {
            'estimator__hidden_units': [[10,20,10],[5,10,5],[100,10],[10,10,10,10]],
            'estimator__learning_rate': [0.1, 0.25, 0.5, 1., 2. ],
            'estimator__optimizer': ['SGD','Adam','Adagrad'],
            'estimator__batch_size': [16,32,64],
            }

class DNNR():
    estimator=MC(learn.TensorFlowDNNClassifier(hidden_units=[10, 20, 10], n_classes=6, steps=1000,verbose=0))
    parameter = {
            'estimator__hidden_units': [[10,20,10],[5,10,5],[100,10],[10,10,10,10]],
            'estimator__learning_rate': [0.1, 0.25, 0.5, 1., 2. ],
            'estimator__optimizer': ['SGD','Adam','Adagrad'],
            'estimator__batch_size': [16,32,64],
            }
