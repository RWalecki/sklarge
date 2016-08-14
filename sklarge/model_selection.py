import os
import shutil
import numpy as np
import pandas as pd
import multiprocessing
import subprocess
import glob
import dill, gzip, h5py
from collections import defaultdict

from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import LabelKFold, LeaveOneLabelOut
from .metrics import mse, pcc

dir_pwd = (os.path.abspath(__file__).rsplit('/',1)[0])


def Eval(path, verbose = 1):
    if verbose:
        print('jobs:',len(glob.glob(path+'/*/setting.dlz')))
        print('done:',len(glob.glob(path+'/*/results.csv')))
    avr_res = defaultdict(list)
    uniq_para = {}
    for f in glob.glob(path+'/*/results.csv'):
        dat = dill.load(open(f.rsplit('/',1)[0]+'/setting.dlz','rb'))
        res = np.genfromtxt(f.rsplit('/',1)[0]+'/results.csv',dtype=str,delimiter=',')
        if len(res.shape)==1:res = res[None,:]

        index = res[:,0]
        res = np.float16(res[:,1:])
        avr_res[str(dat['para'])].append(res)
        uniq_para[str(dat['para'])]=dat['para']

    # # get tables with results
    paras, table = [],[]
    for para in avr_res:
        avr_res[para] = np.mean(np.stack(avr_res[para]),0)
        table.append(avr_res[para])
        paras.append(para)

    table = np.stack(table).transpose(1,0,2)

    # results of first metric in table [para X output]
    idx = np.argmax(table[0],0)

    best_params_ = uniq_para[paras[np.unique(idx)[0]]]

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
    '''
    '''
    def __init__(
            self, 
            estimator, 
            param_grid = 'default', 
            scoring = None,
            n_jobs = -1, 
            cv=None, 
            cv_folds = 'all',
            refit=False,
            save_pred = False,
            verbose=0,
            out_path = '.tmp'
            ):
        '''
        '''
        self.estimator = estimator
        self.param_grid = param_grid
        self.n_jobs = n_jobs
        self.cv = cv
        self.cv_folds = cv_folds
        self.verbose = verbose
        self.refit = refit
        self.scoring = scoring
        self.save_pred = save_pred
        self.out_path = out_path

    def _to_h5_string(self, X,y, labels=None):
        '''
        '''
        # if not labels/subjects, give each sample unique label
        if labels==None:labels = np.arange(X.shape[0])

        _args = []
        for dset, name in zip([X,y,labels],['X','y','labels']):

            # converte to hdf5 if file is in numpy format
            if type(dset) is not h5py._hl.dataset.Dataset:
                pwd = self.out_path+'.tmp/'+name+'.h5'
                try:
                    os.makedirs(pwd.rsplit('/',1)[0])
                except:
                    pass
                os.remove(pwd) if os.path.exists(pwd) else None
                with h5py.File(pwd) as f:
                    h5_dset = f.create_dataset(name,data=dset)
                    id = [h5_dset.file.filename,name]
            else:
                id = [dset.file.filename,dset.name]
            _args.append(id)

        # return paths to files and datasts
        return _args

    def fit(self, X, y, labels=None, submit='local'):
        '''
        '''

        if self.verbose:print('fitting ...')
        self.out_path = os.path.abspath(self.out_path)
        if self.out_path[-1] is not '/': self.out_path+='/'
        shutil.rmtree(self.out_path, ignore_errors=True)

        if self.verbose:print('creating hdf5 files ...')
        X, y, labels = self._to_h5_string(X, y, labels)

        if self.verbose:print('creatin job folder ...')
        self._create_jobs(X, y, labels, self.cv, self.out_path)

        if self.verbose:print('running jobs ...')
        if submit=='local':
            self._run_local(self.out_path, self.n_jobs)
        if submit=='condor':
            self._run_condor(self.out_path, self.n_jobs)

    def get_best_param(self):
        best_score_, best_params_, table = Eval(self.out_path,verbose = 0)
        return best_params_

    def get_best_score(self):
        best_score_, best_params, table = Eval(self.out_path,verbose = 0)
        return best_score_

    def _create_jobs(self, X, y, labels, cv, out_path):
        '''
        '''
        if self.cv_folds=='all':
            cv_folds = np.arange(cv.n_folds)
        else:
            cv_folds = self.cv_folds

        if self.param_grid=='default':
            self.param_grid = self.estimator.param_grid
        params = ParameterGrid(self.param_grid)

        if self.scoring!=None:
            scoring=self.scoring
        else:
            scoring=[mse, pcc]
        if type(scoring) is not list:scoring = [scoring]

        if self.verbose:print('n_tasks:',len(params)*len(cv_folds))
        job = 0
        for fold in cv_folds:
            for para in  params:
                if self.verbose>1:print(fold,para)

                out = '/'.join([out_path,str(job)])
                if not os.path.exists(out):os.makedirs(out)
                experiment = {}
                experiment['X']=X
                experiment['y']=y
                experiment['labels']=labels
                experiment['para']=para
                experiment['fold']=fold
                experiment['scoring']=scoring
                experiment['save_pred']=self.save_pred
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
            f.write('executable      = '+out_path+'execute.sh\n')
            f.write('output          = '+out_path+'$(Process)/tmp.out\n')
            f.write('error           = '+out_path+'$(Process)/tmp.err\n')
            f.write('log             = '+out_path+'tmp.log\n')
            f.write('arguments       = $(Process)\n')
            f.write('queue '+n+'\n')

        subprocess.call(['condor_submit',out_path+'/run_condor.cmd'])
