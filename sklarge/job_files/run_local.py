import numpy as np
import dill
import os
import h5py

dir_pwd = (os.path.abspath(__file__).rsplit('/',1)[0])

# open file containing settings for experiment
dat = dill.load(open(dir_pwd+'/setting.dlz','rb'))

X_tr, Y_tr = [], []
for X_pwd, Y_pwd in dat['data_tr']:
    with h5py.File(X_pwd) as h5_file:
        X_tr.append((h5_file['data'][::]))
    with h5py.File(Y_pwd) as h5_file:
        Y_tr.append((h5_file['data'][::]))
X_tr = np.vstack(X_tr)
Y_tr = np.vstack(Y_tr)

X_te, Y_te = [], []
for X_pwd, Y_pwd in dat['data_te']:
    with h5py.File(X_pwd) as h5_file:
        X_te.append((h5_file['data'][::]))
    with h5py.File(Y_pwd) as h5_file:
        Y_te.append((h5_file['data'][::]))
X_te = np.vstack(X_te)
Y_te = np.vstack(Y_te)

if dat['output_labels']!=None:
    Y_te = Y_te[:,dat['output_labels']]
    Y_tr = Y_tr[:,dat['output_labels']]

# load estimator
clf = dat['clf']
clf.set_params(**dat['para'])


try:
    # try to fit an tf_estimater
    clf.fit( X_tr, Y_tr, X_te, Y_te, save=dir_pwd+'/model_output' )
except TypeError:
    clf.fit( X_tr, Y_tr)


# save results (validation)
Y_hat_te = clf.predict( X_te )

if dat['save_pred']:
    np.savez(dir_pwd+'/y_hat_te', Y_hat_te)
    np.savez(dir_pwd+'/y_te', Y_te)

names,score = [],[]
for scoring in dat['scoring']:
    names.append(scoring._score_func.__name__)
    score.append(np.array([scoring._score_func(Y_hat_te,Y_te)*scoring._sign]))

names = np.vstack(names)
score = np.vstack(score)
table = np.hstack((names,score))
np.savetxt(dir_pwd+'/results_te.csv', table, fmt="%s",delimiter=',')

# save results (training)
Y_hat_tr = clf.predict( X_tr )
if dat['save_pred']:
    np.savez(dir_pwd+'/y_hat_tr.h5', Y_hat_tr)
    np.savez(dir_pwd+'/y_tr', Y_tr)

names,score = [],[]
for scoring in dat['scoring']:
    names.append(scoring._score_func.__name__)
    score.append(np.array([scoring._score_func(Y_hat_tr,Y_tr)*scoring._sign]))

names = np.vstack(names)
score = np.vstack(score)
table = np.hstack((names,score))
np.savetxt(dir_pwd+'/results_tr.csv', table, fmt="%s",delimiter=',')
