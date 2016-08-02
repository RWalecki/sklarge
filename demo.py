import multioutput_evaluation as me
from sklearn import model_selection
import numpy as np
import os

import h5py
f = h5py.File('tests/data/test.h5')

# cv = model_selection.LeaveOneLabelOut()
cv = model_selection.LabelKFold(2)
clf = me.sk_estimator.MVR()

# simple fit/prediction
clf.fit(f['X'],f['y'])
y_hat = clf.predict(f['X'])

# apply tr/te split
clf.fit(f['X'],f['y'],np.arange(0,50))
y_hat = clf.predict(f['X'],np.arange(50,100))

# # run the same but with numpy arrays as input data
clf.fit(f['X'][:],f['y'][:])
y_hat = clf.predict(f['X'][:])

# apply parameter search
GS = me.GridSearchCV(
        clf=clf,
        cv=cv,
        verbose = 2,
        )

GS.fit(
        X = '/vol/hmi/projects/robert/data/CNN_DATA/data_gray/disfa.h5/points',
        y = '/vol/hmi/projects/robert/data/CNN_DATA/data_gray/disfa.h5/au_int',
        labels = '/vol/hmi/projects/robert/data/CNN_DATA/data_gray/disfa.h5/subject_id',
        tmp = 'tmp',
        submit='local',
        )

print(clf.__class__.__name__)
GS.eval('tmp/'+clf.__class__.__name__)

# todo:
# wait untill condor finishes
# implement a wait/status function
