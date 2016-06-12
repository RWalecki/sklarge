import os
import shutil
import itertools
import multiprocessing
import glob
import subprocess
import inspect

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid
from metrics import pcc, icc, mse, f1_detection 
import cPickle
import dill
import gzip
import h5py



class GridSearchCV():

    def __init__(self, estimator, param_grid, scoring=pcc , n_jobs = -1, cv=None, verbose=0, mode='w', output='/tmp'):
        self.estimator = estimator
        self.param_grid = param_grid
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.cv = cv
        self.verbose = verbose
        self.mode=mode

        if output[-1]=='/':output=output[:-1]
        self.output=output

    def fit(self, X, y,labels=None):
        params = ParameterGrid(self.param_grid)
        if self.mode=='w':shutil.rmtree(self.output, ignore_errors=True)

        # ToDo: check if this can be done cleaner 
        with h5py.File(X.rsplit('/',1)[0]) as f:
            labels=f[labels.rsplit('/',1)[1]][::]
            y_lab = f[y.rsplit('/',1)[1]][::]

            data_splits = [i for i in self.cv.split(labels,labels,labels)]
            self._create_jobs(X, y, data_splits, params, self.output)
            self._run_local(self.output,self.n_jobs,self.mode,self.verbose)
            y_hat = self._joint_resutls(self.output)

            # load labels and evaluate models
            tab, y_best, para_best = self._find_best_performing_parameter(y_lab, y_hat, self.scoring, 1)
            pd.set_option('display.float_format', lambda x: '%.2f' % x)
            if self.verbose>0:print tab
            if self.verbose>1:
                for i,p in enumerate(para_best):
                    print i,p

    def _create_jobs(self, X, y, data_splits, params, out_path):
        if self.verbose:print 'n_tasks:',len(params)*len(data_splits)
        for i,[param, data_split] in enumerate(itertools.product(params,data_splits)):
            job_id = str(i).zfill(6)
            out = '/'.join([out_path,job_id])
            if not os.path.exists(out):os.makedirs(out)
            experiment = {}
            experiment['pwd_X']=X
            experiment['pwd_y']=y
            experiment['data_split']=data_split
            experiment['param']=param
            experiment['estimator']=self.estimator
            dill.dump(experiment, file(out+'/setting.dlz','wb'))

    def _run_local(self,path,n_jobs=-1,mode='c',verbose=1):
        if n_jobs==-1:n_jobs=multiprocessing.cpu_count()
        py_file = '/'.join([inspect.getfile(inspect.currentframe()).rsplit('/',1)[0],'run_local.py'])
        subprocess.call(["python", py_file, path, str(mode), str(n_jobs),str(verbose)])

    def _find_best_performing_parameter(self, y_lab, y_hat, metric=pcc, independent=True):
        '''
        '''
        params = y_hat.keys()
        res = np.vstack([metric(y_lab,y_hat[p]) for p in params])
        if independent:
            idx = np.argmax(res,0)
        else:
            idx = np.tile(np.argmax(res.mean(1)),(res.shape[1]))
        param = [params[i] for i in idx]

        y_best = np.array([y_hat[params[i]][:,n] for n,i in enumerate(idx)]).T
        dat = np.vstack([
            pcc(y_lab,y_best),icc(y_lab,y_best),mse(y_lab,y_best),f1_detection(y_lab,y_best)
            ])
        dat = np.hstack([dat,dat.mean(1)[:,None]])
        columns = [str(i) for i in np.arange(y_lab.shape[1])]+['avr.']
        index = ['PCC','ICC','MSE','F1']
        tab = pd.DataFrame(dat,index=index, columns = columns)
        return tab, y_best, param

    def _joint_resutls(self,output):
        dat = dill.load(file(glob.glob(output+'/*/setting.dlz')[0],'rb'))
        y_hat  = dill.load(gzip.open(glob.glob(output+'/*/predictions.dlz')[0],'rb'))
        N = len(dat['data_split'][0])+len(dat['data_split'][1])
        M = y_hat.shape[1]

        # initialize results for each setting
        results = {}
        for f in glob.glob(output+'/*/setting.dlz'):
            dat = dill.load(file(f,'rb'))
            key = str(dat['param'])
            results[key]=np.zeros((N,M))

    
        # fill with predictions
        for f in glob.glob(output+'/*/predictions.dlz'):
            pwd = f.rsplit('/',1)[0]
            y_hat   = dill.load(gzip.open(pwd+'/predictions.dlz','rb'))
            dat = dill.load(file(pwd+'/setting.dlz','rb'))
            te = dat['data_split'][1]
            key = str(dat['param'])
            results[key][te]=y_hat

        return results
