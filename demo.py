from sklearn.datasets import make_multilabel_classification
import multioutput_evaluation as me
import numpy as np

# generate multilabel dataset 
X,y = make_multilabel_classification(n_classes=10,random_state=0)

# 10 subjects
S = np.random.randint(0,10,y.shape[0])

# run X fold subject independent corss validation
clf = me.sk_estimator.MVR()
me.run(clf, X, y, S, folds=3, out='/tmp')
