import os
import glob
import shutil
import multiprocessing
from copy import deepcopy
import pandas as pd
pd.set_option('display.float_format', lambda x: '%.2f' % x)

import numpy as np
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.model_selection import ParameterGrid
from metrics import pcc, icc, mse, f1_detection 



def _predict(args):
    clf, X, y, S, cv, out, setting = args

    name = clf.__class__.__name__
    id = str(np.abs(hash(str(setting)))).zfill(19)
    pwd_out = out+name+'/'+id
    if os.path.isfile(pwd_out+'.npz'):
        dat = np.load(pwd_out+'.npz')
        return dat['y_hat']

    _clf = deepcopy(clf.estimator)
    _clf.set_params(**setting)
    y_hat = model_selection.cross_val_predict(_clf,X,y,S,cv)

    np.savez_compressed(pwd_out,y_hat=y_hat, y=y, setting=setting)

    return y_hat

def _evaluate( pwd, metric=pcc, independent=True):
    '''
    '''
    Y_HAT, SETTING = [], []
    for f in glob.glob(pwd+'/*.npz'):
        dat = np.load(f)
        y_hat = dat['y_hat']
        y = dat['y']
        setting = dat['setting']
        Y_HAT.append(y_hat)
        SETTING.append(setting)

    
    res = np.vstack([metric(ii,y) for ii in Y_HAT])
    if independent:
        idx = np.argmax(res,0)
    else:
        idx = np.tile(np.argmax(res.mean(1)),(res.shape[1]))
    setting = [SETTING[ii] for ii in idx]

    y_hat = np.array([Y_HAT[i][:,n] for n,i in enumerate(idx)]).T
    dat = np.vstack([
        pcc(y_hat,y),icc(y_hat,y),mse(y_hat,y),f1_detection(y_hat,y)
        ])
    dat = np.hstack([dat,dat.mean(1)[:,None]])
    columns = [str(i) for i in np.arange(y_hat.shape[1])]+['avr.']
    index = ['PCC','ICC','MSE','F1']
    tab = pd.DataFrame(dat,index=index, columns = columns)
    return tab, y_hat, setting

def run(clf, X, y, S, folds=5, out='/tmp/', mode='w', verbose=1):
    '''
    '''
    name = clf.__class__.__name__
    if out[-1]!='/':out+='/'
    if mode=='w':shutil.rmtree(out+name, ignore_errors=True)
    if not os.path.exists(out+name):os.makedirs(out+name)

    folds = min([folds,len(np.unique(S))])
    cv = model_selection.LabelKFold(folds)

    scaler = preprocessing.StandardScaler()
    X = scaler.fit(X).transform(X)

    p = multiprocessing.Pool(12)

    args = []
    for setting in model_selection.ParameterGrid(clf.parameter):
        args.append([clf, X, y, S, cv, out, setting])
    y_hat = np.stack(p.map(_predict,args))
    p.close()

    tab, y_hat, setting = _evaluate(out+name)
    if verbose>0:
        print name
        print tab
        print
    if verbose>1:
        for i,s in enumerate(setting):print i,s
    return y_hat
