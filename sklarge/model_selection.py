import os
import shutil
import numpy as np
import pandas as pd
pd.set_option('expand_frame_repr', False)
import multiprocessing
import subprocess
import glob
import dill, gzip, h5py
from collections import defaultdict
from random import shuffle

from sklearn.model_selection import ParameterGrid
from .metrics import mse, mae, icc, pcc, acc

dir_pwd = (os.path.abspath(__file__).rsplit('/',1)[0])




class GridSearchCV():
    '''
    '''
    def __init__(
            self, 
            estimator, 
            param_grid = 'default', 
            scoring = None,
            save_pred = False,
            verbose=0,
            out_path = '.tmp'
            ):
        '''
        '''
        self.estimator = estimator
        self.param_grid = param_grid
        self.verbose = verbose
        self.scoring = scoring
        self.save_pred = save_pred
        self.out_path = os.path.abspath(out_path)

    def make_jobs(self, X, y, idx,  out_path='/tmp/GridSearchCV'):
        '''
        '''
        assert(len(X)==len(y)),'features and labels have not the same length'

        if self.verbose:print('fitting ...')
        self.out_path = os.path.abspath(self.out_path)
        if self.out_path[-1] is not '/': self.out_path+='/'
        shutil.rmtree(self.out_path, ignore_errors=True)
        if not os.path.exists(out_path):os.makedirs(out_path)

        if type(X[0])!=str:
            X = self._make_to_h5(X, self.out_path+'/.tmp_X')
        if type(y[0])!=str:
            y = self._make_to_h5(y, self.out_path+'/.tmp_y')

        if self.verbose:print('creating job folder ...')
        self._create_jobs(X, y, idx, self.out_path)

        # if self.verbose:print('running jobs ...')
        # if submit=='local':
            # self._run_local(self.out_path, self.n_jobs)
        # if submit=='condor':
            # self._run_condor(self.out_path, self.n_jobs)

    def _make_to_h5(self, data_seq, path):
        h5_path = []
        for i, data in enumerate(data_seq):
            path_data_set = path+str(i).zfill(6)+'.h5'
            with h5py.File(path_data_set) as h5_file:
                h5_file.create_dataset('data', data=data)
                h5_path.append(path_data_set)

        return h5_path

    def get_best_param(self):
        best_score_, best_params_, table = Eval(self.out_path,verbose = 0)
        return best_params_

    def get_best_score(self):
        best_score_, best_params, table = Eval(self.out_path,verbose = 0)
        return best_score_

    def _create_jobs(self, X, y, idx, out_path):
        '''
        '''

        if self.param_grid=='default':
            self.param_grid = self.estimator.param_grid
        params = ParameterGrid(self.param_grid)
        params = [i for i in params]
        shuffle(params)


        if self.scoring==None:
            scoring=[acc, mse, mae, pcc, icc]
        else:
            scoring=self.scoring

        if type(scoring) is not list:scoring = [scoring]

        if self.verbose:print('folds:'.ljust(10),len(X))
        if self.verbose:print('parameter:'.ljust(10),len(X))
        if self.verbose:print('n_tasks:'.ljust(10),len(params)*len(X))


        job = 0
        for f,[tr,te] in enumerate(idx):
            data_tr = [[X[i], y[i]] for i in tr]
            data_te = [[X[i], y[i]] for i in te]

            for para in  params:
                if self.verbose>1:
                    print( str(job).ljust(3), str(f).ljust(2), para )
                out = '/'.join([out_path,str(job)])
                if not os.path.exists(out):os.makedirs(out)
                experiment = {}
                experiment['data_tr'] = data_tr
                experiment['data_te'] = data_te
                experiment['save_pred'] = self.save_pred 
                experiment['para'] = para
                experiment['scoring'] = scoring
                experiment['clf'] = self.estimator

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
            f.write('executable      = '+out_path+'execute.sh\n')
            f.write('output          = '+out_path+'$(Process)/tmp.out\n')
            f.write('error           = '+out_path+'$(Process)/tmp.err\n')
            f.write('log             = '+out_path+'tmp.log\n')
            f.write('arguments       = $(Process)\n')
            f.write('queue '+n+'\n')

        subprocess.call(['condor_submit',out_path+'/run_condor.cmd'])

def run_local(pwd, n_jobs = -1):
    if n_jobs==-1:n_jobs=multiprocessing.cpu_count()
    p = multiprocessing.Pool(n_jobs)

    jobs = glob.glob(pwd+'/*/run_local.py')
    jobs = [i for i in zip(['python']*len(jobs),jobs)]

    p.map(subprocess.call,jobs)
    p.close()
