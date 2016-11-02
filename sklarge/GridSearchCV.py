import os
import shutil
import numpy as np
import glob
import dill
import h5py
from random import shuffle

from sklearn.model_selection import ParameterGrid
from .metrics import rmse, mae, icc, pcc, acc

dir_pwd = (os.path.abspath(__file__).rsplit('/',1)[0])

def _to_h5(data_seq, path):
    h5_path = []
    for i, data in enumerate(data_seq):
        path_data_set = path+str(i).zfill(6)+'.h5'
        with h5py.File(path_data_set) as h5_file:
            h5_file.create_dataset('data', data=data)
            h5_path.append(path_data_set)

    return h5_path


class GridSearchCV():
    '''
    '''
    def __init__(
            self, 
            estimator, 
            param_grid = 'default', 
            output_labels = None,
            scoring=[acc, rmse, mae, pcc, icc],
            save_pred = False,
            verbose=0,
            ):
        '''
        '''
        self.estimator = estimator
        self.param_grid = param_grid
        self.verbose = verbose
        self.scoring = scoring
        self.output_labels = output_labels
        self.save_pred = save_pred


    def _create_job_files(self, X, y, idx,  out_path='/tmp/GridSearchCV',mode='r'):
        '''
        '''
        assert(len(X)==len(y)),'features and labels have not the same length'

        out_path = os.path.abspath(out_path)
        if mode=='w':
            shutil.rmtree(out_path, ignore_errors=True)
        if not os.path.exists(out_path):os.makedirs(out_path)

        if type(X[0])!=str:
            X = _to_h5(X, out_path+'/.tmp_X')
        if type(y[0])!=str:
            y = _to_h5(y, out_path+'/.tmp_y')

        if self.param_grid=='default':
            self.param_grid = self.estimator.param_grid
        params = ParameterGrid(self.param_grid)
        params = [i for i in params]
        shuffle(params)

        if self.verbose:print('folds:'.ljust(10),len(idx))
        if self.verbose:print('parameter:'.ljust(10),len(idx))
        if self.verbose:print('n_tasks:'.ljust(10),len(params)*len(X))


        job = 0
        for f,[tr,te] in enumerate(idx):
            for para in  params:
                if self.verbose>1:print( str(job).ljust(3), str(f).ljust(2), para )

                experiment = {
                    'data_tr' : [[X[i], y[i]] for i in tr],
                    'data_te' : [[X[i], y[i]] for i in te],
                    'save_pred' : self.save_pred ,
                    'output_labels' : self.output_labels,
                    'scoring' : self.scoring,
                    'clf' : self.estimator,
                    'para' : para,
                }

                out = '/'.join([out_path,str(job)])

                os.makedirs(out)
                dill.dump(experiment, open(out+'/setting.dlz','wb'))
                shutil.copy(dir_pwd+'/job_files/run_local.py',out)
                shutil.copy(dir_pwd+'/job_files/execute.sh',out_path)

                job+=1
