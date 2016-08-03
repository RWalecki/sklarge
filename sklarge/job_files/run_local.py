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
f_X = h5py.File(dat['X'].rsplit('/',1)[0])
X = f_X[dat['X'].rsplit('/',1)[1]]

# open dataset: targets
f_y = h5py.File(dat['y'].rsplit('/',1)[0])
y = f_y[dat['y'].rsplit('/',1)[1]]

# open dataset: labels
f_labels = h5py.File(dat['labels'].rsplit('/',1)[0])
labels = f_labels[dat['labels'].rsplit('/',1)[1]]

# training/test split
tr, te = [i for i in dat['cv'].split(labels,labels,labels)][dat['fold']]

# check if fit on h5 subset is implemented
if 'idx' in inspect.getargspec(clf.fit).args:
    clf.fit( X, y, h5_idx=tr )
    y_hat = clf.predict( X, te )
else:
    clf.fit( X[tr.tolist()], y[tr.tolist()] )
    y_hat = clf.predict( X[te.tolist()] )

names,score = [],[]
for scoring in dat['scoring']:
    names.append(scoring._score_func.func_name)
    score.append(np.array([scoring._score_func(y[te.tolist()],y_hat)*scoring._sign]))

names = np.vstack(names)
score = np.vstack(score)
table = np.hstack((names,score))
np.savetxt(dir_pwd+'/results.csv', table, fmt="%s",delimiter=',')

f_X.close()
f_y.close()
f_labels.close()
