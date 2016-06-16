'''
example:
python run_local.py <path to job folder>
python run_local.py /tmp/blabla/ w 1
'''

import dill, pickle, gzip
import h5py 
import os
dir_pwd = (os.path.abspath(__file__).rsplit('/',1)[0])


# open file that contains parameter for the experiment
dat = dill.load(open(dir_pwd+'/setting.dlz','rb'))
with h5py.File(dat['pwd_X'].rsplit('/',1)[0]) as f:

    # load data from root hdf5 file
    X = f[dat['pwd_X'].rsplit('/',1)[1]]
    y = f[dat['pwd_y'].rsplit('/',1)[1]]
    tr, te = dat['data_split']
    clf = dat['estimator']
    clf.set_params(**dat['param'])
    clf.fit(X,y,tr)
    y_hat = clf.predict(X,te)

    # store predictions
    pickle.dump(y_hat,gzip.open(dir_pwd+'/predictions.dlz','wb'))
    print('__done__')
