from sklearn.datasets import make_multilabel_classification
import multioutput_evaluation as me
import numpy as np

# generate multilabel dataset 
X,y = make_multilabel_classification()

# 10 subjects
S = np.random.randint(0,10,y.shape[0])

clf = me.estimator.MVR()
me.benchmark(clf, X, y, S, folds=3, out='/tmp')
