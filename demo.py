import multioutput_evaluation as me
from sklearn import model_selection
import numpy as np
import os

import h5py
f = h5py.File('tests/data/test.h5')

cv = model_selection.LeaveOneLabelOut()
# cv = model_selection.LabelKFold(2)
CLF=[]
CLF.append(me.sk_estimator.MVR())
CLF.append(me.sk_estimator.MTL())
CLF.append(me.sk_estimator.SVC())
CLF.append(me.sk_estimator.SVR())


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
for clf in CLF:
    GS = me.GridSearchCV(
            clf=clf,
            cv=cv,
            verbose = 2,
            )

    GS.fit(
            X = '/vol/hmi/projects/robert/data/CNN_DATA/data_gray/disfa.h5/points',
            y = '/vol/hmi/projects/robert/data/CNN_DATA/data_gray/disfa.h5/au_int',
            labels = '/vol/hmi/projects/robert/data/CNN_DATA/data_gray/disfa.h5/subject_id',
            tmp = 'tmp/aaa',
            submit='condor',
            )

    # print(clf.__class__.__name__)
    # GS.eval('tmp/aaa/'+clf.__class__.__name__)

# todo:
# wait untill condor finishes
# implement a wait/status function

