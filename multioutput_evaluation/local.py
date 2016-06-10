import os
import shutil
from copy import deepcopy
import pandas as pd
import itertools
import pprint
pp = pprint.PrettyPrinter(indent=2)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

import numpy as np
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.model_selection import ParameterGrid
from metrics import pcc, icc, mse, f1_detection 

def _predict(clf,X,y,S,cv,out,setting):
    # id = str(np.abs(hash(str(setting)))).zfill(19)
    # pwd_out = out+id
    # if os.path.isfile(pwd_out+'.npz'):
        # dat = np.load(pwd_out+'.npz')
        # return dat['y_hat']

    _clf = deepcopy(clf.estimator)
    _clf.set_params(**setting)
    y_hat = model_selection.cross_val_predict(_clf,X,y,S,cv)
    # np.savez_compressed(pwd_out,y_hat=y_hat, y=y, setting=setting)

    return y_hat

def _find_best_performing_setting( y, y_hat, settings, metric=pcc, independent=True):
    '''
    '''
    res = np.vstack([metric(ii,y) for ii in y_hat])
    if independent:
        idx = np.argmax(res,0)
    else:
        idx = np.tile(np.argmax(res.mean(1)),(res.shape[1]))
    setting = [settings[ii] for ii in idx]

    y_best = np.array([y_hat[i][:,n] for n,i in enumerate(idx)]).T
    dat = np.vstack([
        pcc(y_best,y),icc(y_best,y),mse(y_best,y),f1_detection(y_best,y)
        ])
    dat = np.hstack([dat,dat.mean(1)[:,None]])
    columns = [str(i) for i in np.arange(y.shape[1])]+['avr.']
    index = ['PCC','ICC','MSE','F1']
    tab = pd.DataFrame(dat,index=index, columns = columns)
    return tab, y_best, setting

def run(clf, X, y, S, cv, metric=pcc, n_jobs=-1, independent=True, out='/tmp/', mode='w', verbose=1):
    '''
    '''
    name = clf.__class__.__name__
    if out[-1]!='/':out+='/'
    if mode=='w':shutil.rmtree(out+name, ignore_errors=True)
    if n_jobs==-1:n_jobs=multiprocessing.cpu_count()
    if not os.path.exists(out):os.makedirs(out)

    settings = model_selection.ParameterGrid(clf.param_grid)
    y_hat = [_predict(clf,X,y,S,cv,out,ii) for ii in settings]

    tab, y_best, setting = _find_best_performing_setting(y, y_hat, settings)
    if verbose>0:
        print name
        print tab
        print
    if verbose>1:
        for i,s in enumerate(setting):
            print 'output:',i
            pp.pprint(s)
    return y_best
