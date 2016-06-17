import dill, pickle, gzip
import h5py 
import os
import numpy as np
dir_pwd = (os.path.abspath(__file__).rsplit('/',1)[0])


# open file that contains parameter for the experiment
dat = dill.load(open(dir_pwd+'/setting.dlz','rb'))
with h5py.File(dat['X'].rsplit('/',1)[0]) as f:

    # load data from root hdf5 file
    X = f[dat['X'].rsplit('/',1)[1]]
    y = f[dat['y'].rsplit('/',1)[1]]
    labels = f[dat['labels'].rsplit('/',1)[1]]

    cv = dat['cv']
    clf = dat['clf']
    clf.set_params(**dat['para'])

    tr,te = [i for i in cv.split(labels,labels,labels)][dat['fold']]

    clf.fit(X,y,tr)

    y_hat = clf.predict(X,te)

    res = np.vstack([metric(y_hat,y[te.tolist()]) for metric in dat['eval']])
    np.savetxt(dir_pwd+'/results.csv', res, delimiter=',')
