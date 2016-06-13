'''
example:
python run_local.py <path to job folder> <continue> <n_jobs> <verbose>
python run_local.py /tmp/blabla/ w 1 1
'''
import sys
import dill, cPickle, gzip
import glob
import h5py 
import os
import multiprocessing
import numpy as np
from sklearn.linear_model import Ridge

def _run_job(args):

    # check if predictions are already computed
    mode = args[1]
    output = args[0].rsplit('/',1)[0]+'/predictions.dlz'
    if args[1]=='c':
        if os.path.isfile(output):return 0

    # open file that contains parameter for the experiment
    dat = dill.load(file(args[0],'rb'))
    with h5py.File(dat['pwd_X'].rsplit('/',1)[0]) as f:

        # load data from root hdf5 file
        X = f[dat['pwd_X'].rsplit('/',1)[1]]
        y = f[dat['pwd_y'].rsplit('/',1)[1]]
        tr,te = dat['data_split']
        clf = dat['estimator']
        clf.set_params(**dat['param'])
        clf.fit(X,y,tr)
        y_hat = clf.predict(X,te)

        # store predictions
        cPickle.dump(y_hat,gzip.open(output,'wb'))

if __name__ == '__main__':
    pwd, mode, n_jobs, verbose = sys.argv[1:5]
    n_jobs = int(n_jobs)
    tasks = glob.glob(pwd+'/*/setting.dlz')
    tasks = zip(tasks,[mode]*len(tasks))

    if n_jobs==1:
        for i,task in enumerate(tasks):
            if verbose:print i
            _run_job(task)
    else:
        p = multiprocessing.Pool(12)
        p.map(_run_job, tasks)
