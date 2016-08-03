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
from multioutput_evaluation.metrics import mse
import pickle as cPickle
import dill
import gzip
import h5py

def Eval(path, verbose = 1):
    if verbose:
        print('jobs:',len(glob.glob(path+'/*/setting.dlz')))
        print('done:',len(glob.glob(path+'/*/results.csv')))
    avr_res = defaultdict(list)
    for f in glob.glob(path+'/*/results.csv'):
        dat = dill.load(open(f.rsplit('/',1)[0]+'/setting.dlz','rb'))
        res = np.genfromtxt(f.rsplit('/',1)[0]+'/results.csv',dtype=str,delimiter=',')
        if len(res.shape)==1:res = res[None,:]

        index = res[:,0]
        res = np.float16(res[:,1:])
        avr_res[str(dat['para'])].append(res)

    # # get tables with results
    paras, table = [],[]
    for para in avr_res:
        avr_res[para] = np.mean(np.stack(avr_res[para]),0)
        table.append(avr_res[para])
        paras.append(para)

    table = np.stack(table).transpose(1,0,2)

    # results of first metric in table [para X output]
    idx = np.argmax(table[0],0)

    best_params_ = paras[np.unique(idx)[0]]

    # # store all results in one table [measures X output]
    dat = np.vstack([tab_[idx,np.arange(idx.shape[0])] for tab_ in table])

    # # add avr for each measure
    dat = np.hstack((dat,dat.mean(1)[:,None]))
    best_score_ = dat[0,-1]
    
    columns = [str(i) for i in np.arange(idx.shape[0])]+['avr.']
    tab = pd.DataFrame(dat,index=index, columns = columns)

    pd.set_option('display.float_format', lambda x: '%.2f' % x)
    if verbose:print(tab)
    return best_score_, best_params_, table


class GridSearchCV():

    def __init__(
            self, 
            estimator, 
            param_grid = 'default', 
            scoring = None,
            n_jobs = -1, 
            cv=None, 
            refit=False,
            verbose=0,
            ):
        '''
        '''
        self.estimator = estimator
        self.param_grid = param_grid
        self.n_jobs = n_jobs
        self.cv = cv
        self.verbose = verbose
        self.refit = refit
        self.scoring = scoring

    def _to_h5(self, X,y,labels=None):
        if labels==None:labels = np.arange(X.shape[0])

        _args = []
        for dset,name in zip([X,y,labels],['X','y','labels']):
            if type(dset) is not h5py._hl.dataset.Dataset:

                pwd = '/tmp/GridSearchCV_tmp_'+name+'.h5'
                os.remove(pwd) if os.path.exists(pwd) else None
                f = h5py.File(pwd)

                f.create_dataset(name,data=dset)
                f.close()
                f = h5py.File(pwd)
                _args.append(f[name])

            else:
                _args.append(dset)
        return _args

    def fit(self, X, y,labels=None, tmp='/tmp/GridSearchCV', submit='local'):
        '''
        '''

        if tmp[-1]!='/':tmp=tmp+'/'
        self.out_path = tmp+self.estimator.__class__.__name__+'/'
        shutil.rmtree(self.out_path, ignore_errors=True)

        X, y, labels = self._to_h5(X, y, labels)


        data_splits = [i for i in self.cv.split(labels[::],labels[::],labels[::])]
        n_folds = len(data_splits)


        self._create_jobs(X, y, labels, n_folds, self.cv, self.out_path)

        if submit=='local':
            self._run_local(self.out_path, self.n_jobs)
        if submit=='condor':
            self._run_condor(self.out_path, self.n_jobs)

    def get_best_param(self):
        best_score_, best_params, table = Eval(self.out_path,verbose = 0)
        return best_params

    def get_best_score(self):
        best_score_, best_params, table = Eval(self.out_path,verbose = 0)
        return best_score_


    def _create_jobs(self, X, y, l, n_folds, cv, out_path):

        '''
        '''
        if self.param_grid=='default':
            self.param_grid = self.estimator.param_grid
        params = ParameterGrid(self.param_grid)

        if self.scoring!=None:
            scoring=self.scoring
        else:
            scoring=[mse]
        if type(scoring) is not list:scoring = [scoring]

        if self.verbose:print('n_tasks:',len(params)*n_folds)
        job = 0
        for fold in range(n_folds):
            for para in  params:

                out = '/'.join([out_path,str(job)])
                if not os.path.exists(out):os.makedirs(out)
                experiment = {}
                experiment['X']=X.file.filename+X.name
                experiment['y']=y.file.filename+y.name
                experiment['labels']=l.file.filename+l.name
                experiment['para']=para
                experiment['fold']=fold
                experiment['scoring']=scoring
                experiment['cv']=cv
                experiment['clf']=self.estimator
                dill.dump(experiment, open(out+'/setting.dlz','wb'))
                shutil.copy(dir_pwd+'/job_files/run_local.py',out)
                shutil.copy(dir_pwd+'/job_files/execute.sh',out_path)

                job+=1

    @staticmethod
    def _run_local(out_path, n_jobs=-1):
        '''
        '''
        # run all jobs on the local machine

        if n_jobs==-1:n_jobs=multiprocessing.cpu_count()
        p = multiprocessing.Pool(n_jobs)

        jobs = glob.glob(out_path+'/*/run_local.py')
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
