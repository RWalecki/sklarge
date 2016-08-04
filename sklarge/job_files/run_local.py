import dill, pickle, gzip, h5py 
import os
import numpy as np
import inspect
dir_pwd = (os.path.abspath(__file__).rsplit('/',1)[0])

# open file that contains parameter for the experiment
dat = dill.load(open(dir_pwd+'/setting.dlz','rb'))

# load estimator
clf = dat['clf']
clf.set_params(**dat['para'])

# open dataset: features
X = h5py.File(dat['X'][0])[dat['X'][1]]
y = h5py.File(dat['y'][0])[dat['y'][1]]
labels = h5py.File(dat['labels'][0])[dat['labels'][1]]

# training/test split
tr, te = [i for i in dat['cv'].split(labels,labels,labels)][dat['fold']]

# check if fit on h5 subset is implemented
if 'idx' in inspect.getargspec(clf.fit).args:
    clf.fit( X, y, h5_idx=tr )
    y_hat = clf.predict( X, te )
else:
    clf.fit( X[tr.tolist()], y[tr.tolist()] )
    y_hat = clf.predict( X[te.tolist()] )

# save results
names,score = [],[]
for scoring in dat['scoring']:
    names.append(scoring._score_func.__name__)
    score.append(np.array([scoring._score_func(y[te.tolist()],y_hat)*scoring._sign]))

names = np.vstack(names)
score = np.vstack(score)
table = np.hstack((names,score))
np.savetxt(dir_pwd+'/results.csv', table, fmt="%s",delimiter=',')

# save predictions
if dat['save_pred']==True:
    with h5py.File(dir_pwd+'/y_hat.h5') as f:
        f.create_dataset('y_hat',data=y_hat)

# close all files
for group in [X,y,labels]:
    try:
        group.file.close()
    except RuntimeError:
        pass
