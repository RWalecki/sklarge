import multioutput_evaluation as me
from sklearn import model_selection
import numpy as np
import os

import h5py
f = h5py.File('tests/data/test.h5')

# cv = model_selection.LeaveOneLabelOut()
# clf = me.sk_estimator.SVR()
clf = me.sk_estimator.MVR()
# clf = me.tf_estimator.DNN_C(verbose=1,max_iter=1)


# simple fit/prediction
# clf.fit(f['X'],f['y'])
# y_hat = clf.predict(f['X'])

# apply tr/te split
# clf.fit(f['X'],f['y'],np.arange(0,50))
# y_hat = clf.predict(f['X'],np.arange(50,100))

# # run the same but with numpy arrays as input data
# clf.fit(f['X'][:],f['y'][:])
# y_hat = clf.predict(f['X'][:])

# apply parameter search
GS = me.GridSearchCV(
        clf = me.sk_estimator.MVR(),
        cv=model_selection.LabelKFold(2),
        verbose = 2,
        )

GS.fit(
        X = '/vol/hmi/projects/robert/data/CNN_DATA/data_gray/disfa.h5/points',
        y = '/vol/hmi/projects/robert/data/CNN_DATA/data_gray/disfa.h5/au_int',
        labels = '/vol/hmi/projects/robert/data/CNN_DATA/data_gray/disfa.h5/subject_id',
        tmp = 'out/MVR2',
        submit='condor',
        )

# GS.eval('out/MVR2')

# todo:
# wait untill condor finishes
# implement a wait/status function

