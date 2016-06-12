import multioutput_evaluation as me
from sklearn import model_selection
import numpy as np
import os
import h5py

f = h5py.File('tests/data/test.h5')

cv = model_selection.LabelKFold(10)
clf = me.sk_estimator.MVR()


# simple fit/prediction
clf.fit(f['X'],f['y'])
y_hat = clf.predict(f['X'])

# apply tr/te split
clf.fit(f['X'],f['y'],np.arange(0,50))
y_hat = clf.predict(f['X'],np.arange(50,100))

# run the same but with numpy arrays as input data
clf.fit(f['X'][:],f['y'][:])
y_hat = clf.predict(f['X'][:])

# apply parameter search
clf = me.GridSearchCV(
        clf,
        clf.param_grid,
        cv=cv,
        n_jobs=-1,
        verbose = 2,
        output = '/tmp/oo',
        mode = 'c'
        )

clf.fit(
        X = '/vol/hmi/projects/robert/data/CNN_DATA/data_gray/fera2015.h5/points',
        y = '/vol/hmi/projects/robert/data/CNN_DATA/data_gray/fera2015.h5/au_int',
        labels = '/vol/hmi/projects/robert/data/CNN_DATA/data_gray/fera2015.h5/subject_id',
        )
