import shutil
import multiprocessing
import glob
import subprocess
from collections import defaultdict

import os
dir_pwd = (os.path.abspath(__file__).rsplit('/',1)[0])


import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid
from .metrics import pcc, icc, mse, f1_detection 
import pickle as cPickle
import dill
import gzip
import h5py


class GridSearchCV():

    def __init__(self, clf, param_grid='default', n_jobs = -1, cv=None, verbose=0):
        '''
        '''
        self.clf = clf
        self.param_grid = param_grid
        self.n_jobs = n_jobs
        self.cv = cv
        self.verbose = verbose

    def fit(self, X, y,labels=None, tmp='/tmp/', submit='local'):
        '''
        '''
        if tmp[-1]!='/':tmp=tmp+'/'
        tmp = tmp+self.clf.__class__.__name__+'/'
        print(tmp)

        shutil.rmtree(tmp, ignore_errors=True)

        # (GET TOTAL NUMBER OF FOLDS)
        # ToDo: check if this can be done in a better way 
        with h5py.File(X.rsplit('/',1)[0]) as f:
            labels_tmp=f[labels.rsplit('/',1)[1]][::]
            data_splits = [i for i in self.cv.split(labels_tmp,labels_tmp,labels_tmp)]
            n_folds = len(data_splits)

        self._create_jobs(X, y, labels, n_folds, self.cv, tmp)

        if submit=='local':
            self._run_local(tmp, self.n_jobs)
        if submit=='condor':
            self._run_condor(tmp, self.n_jobs)

    def _create_jobs(self,X, y, l, n_folds, cv, out_path):
        '''
        '''
        if self.param_grid=='default':
            self.param_grid = self.clf.param_grid
        params = ParameterGrid(self.param_grid)

        if self.verbose:print('n_tasks:',len(params)*n_folds)
        job = 0
        for fold in range(n_folds):
            for para in  params:

                out = '/'.join([out_path,str(job)])
                if not os.path.exists(out):os.makedirs(out)
                experiment = {}
                experiment['X']=X
                experiment['y']=y
                experiment['labels']=l
                experiment['para']=para
                experiment['fold']=fold
                experiment['eval']=[mse,pcc,icc,f1_detection]
                experiment['cv']=cv
                experiment['clf']=self.clf
                dill.dump(experiment, open(out+'/setting.dlz','wb'))
                shutil.copy(dir_pwd+'/job_files/run_local.py',out)
                shutil.copy(dir_pwd+'/job_files/execute.sh',out_path)

                job+=1

    @staticmethod
    def _run_local(out_path, n_jobs=-1):
        '''
        '''
        # run all jobs on the local machine
        jobs = glob.glob(out_path+'/*/run_local.py')
        if n_jobs==-1:n_jobs=multiprocessing.cpu_count()
        p = multiprocessing.Pool(n_jobs)
        jobs = [i for i in zip(['python']*len(jobs),jobs)]
        p.map(subprocess.call,jobs)
        p.close()

    @staticmethod
    def _run_condor(out_path, n_jobs=-1):
        '''
        '''
        n = str((len(glob.glob(out_path+'/*/setting.dlz'))))

        # create condor file:
        with open(out_path+'/run_condor.cmd','w') as f:
            f.write('executable      = '+out_path+'/execute.sh\n')
            f.write('output          = '+out_path+'/$(Process)/tmp.out\n')
            f.write('error           = '+out_path+'/$(Process)/tmp.err\n')
            f.write('log             = '+out_path+'/tmp.log\n')
            f.write('arguments       = $(Process)\n')
            f.write('queue '+n+'\n')

        subprocess.call(['condor_submit',out_path+'/run_condor.cmd'])

    @staticmethod
    def eval(output='/tmp/GridSearchCV', independent=True):
        '''
        '''
        print('jobs:',len(glob.glob(output+'/*/setting.dlz')))
        print('done:',len(glob.glob(output+'/*/results.csv')))

        avr_res = defaultdict(list)
        for f in glob.glob(output+'/*/results.csv'):
            dat = dill.load(open(f.rsplit('/',1)[0]+'/setting.dlz','rb'))
            res = np.genfromtxt(f.rsplit('/',1)[0]+'/results.csv',delimiter=',')
            avr_res[str(dat['para'])].append(res)

        # get tables with results
        table = []
        for para in avr_res:
            avr_res[para] = np.mean(np.stack(avr_res[para]),0)
            table.append(avr_res[para])
        table = np.stack(table).transpose(1,0,2)

        # results of first metric in table [para X output]
        if independent:
            idx = np.argmin(table[0],0)
        else:
            idx = np.tile(np.argmin(table[0].mean(1)),table.shape[2])

        # store all results in one table [measures X output]
        dat = np.vstack([tab_[idx,np.arange(idx.shape[0])] for tab_ in table])

        # add avr for each measure
        dat = np.hstack((dat,dat.mean(1)[:,None]))
        
        columns = [str(i) for i in np.arange(idx.shape[0])]+['avr.']
        index = ['MSE','PCC','ICC','F1']
        tab = pd.DataFrame(dat,index=index, columns = columns)

        pd.set_option('display.float_format', lambda x: '%.2f' % x)
        print(tab)
