import os
import glob
import shutil
import multiprocessing
from copy import deepcopy
import pandas as pd

import numpy as np
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.model_selection import ParameterGrid
from metrics import pcc, icc, mse



def _predict(args):
    clf, X, y, S, cv, out, setting = args

    name = clf.__class__.__name__
    id = str(np.abs(hash(str(setting)))).zfill(19)
    pwd_out = out+name+'/'+id
    if os.path.isfile(pwd_out+'.npy'):
        dat = np.load(pwd_out+'.npy')
        return dat[0]

    _clf = deepcopy(clf.estimator)
    _clf.set_params(**setting)
    y_hat = model_selection.cross_val_predict(_clf,X,y,S,cv)

    np.save(pwd_out,[y_hat,y,setting])

    return y_hat

def _evaluate( pwd, metric=pcc, independent=True):
    '''
    '''
    Y_HAT,SETTING = [],[]
    for f in glob.glob(pwd+'/*.npy'):
        y_hat, y, setting = np.load(f)
        Y_HAT.append(y_hat)
        SETTING.append(setting)

    # getting best model (per output)
    res = np.vstack([metric(ii,y) for ii in Y_HAT])
    if independent:
        idx = np.argmax(res,0)
    else:
        idx = np.tile(np.argmax(res.mean(1)),(res.shape[1]))

    y_hat = np.array([Y_HAT[i][:,n] for n,i in enumerate(idx)]).T
    print pcc(y_hat,y).mean()
    print icc(y_hat,y).mean()
    print mse(y_hat,y).mean()

def benchmark(clf, X, y, S, folds=5, out='/tmp/', mode='w', verbose=0):
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

    _evaluate(out+name)

        # a1 a2 a3 avr
    # icc
    # pcc
    # mse

