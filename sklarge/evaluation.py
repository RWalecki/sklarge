import glob
import pandas as pd
pd.set_option('display.float_format', lambda x: '%.2f' % x)
from collections import defaultdict
import dill
import numpy as np

pd.set_option('expand_frame_repr', False)

def evaluation(path, best_joint=True, verbose=1, condition={}):

    done_experiments = glob.glob(path+'/*/results.csv')

    if verbose>1:
        print('jobs:',len(glob.glob(path+'/*/setting.dlz')))
        print('done:',len(done_experiments))

    if len(done_experiments)==0:
        print('could not find any results in \n',path)
        return 0

    ################################################################ 
    #  get unique parameter keys
    ################################################################ 
    f = done_experiments[0]
    dat = dill.load(open(f.rsplit('/',1)[0]+'/setting.dlz','rb'))
    par = dat['para']
    keys = [i for i in par.keys()]
    keys.sort()
    ################################################################ 
    
    avr_res_for_all_folds = defaultdict(list)
    paths_for_folds = defaultdict(list)
    parameter_for_folds   = {}
    
    ################################################################ 
    #  get parameter grid 
    ################################################################ 
    for f in done_experiments:

        dat = dill.load(open(f.rsplit('/',1)[0]+'/setting.dlz','rb'))
        par = dat['para']

        skip_parameter = False
        for c in condition:
            if (par[c]!=condition[c]):
                skip_parameter=True
        if skip_parameter:
            continue

        res = np.genfromtxt(f.rsplit('/',1)[0]+'/results.csv',dtype=str,delimiter=',')
        if len(res.shape)==1:res = res[None,:]
        index = res[:,0]
        paths_for_folds[str(par)].append(f)

        # remove the metric names
        res = np.float16(res[:,1:])

        avr_res_for_all_folds[str(par)].append(res)
        parameter_for_folds[str(par)]= par

    # find numper of folds
    n_folds = 0
    for para in avr_res_for_all_folds:
        n_folds = max(n_folds,len(avr_res_for_all_folds[para]))

    # compute averages per parameter setting
    table = []
    para_list = []
    for para in avr_res_for_all_folds:
        res_per_fold  = np.stack(avr_res_for_all_folds[para])

        # if not all experiments are done, skip this setting
        if len(res_per_fold)!=n_folds:continue
        avr_res = np.mean(res_per_fold,0)

        table.append( avr_res )
        para_list.append(para)

    table = np.stack(table).transpose(1,0,2)

    # results of first metric in table [para X output]
    if best_joint==True:
        idx = np.tile(np.argmax(table[-1].mean(1)),table.shape[2])
    else:
        idx = np.argmax(table[-1],0)

    
    best_params = parameter_for_folds[para_list[int(np.median(idx))]]

    # # store all results in one table [measures X output]
    dat = np.vstack([tab_[idx,np.arange(idx.shape[0])] for tab_ in table])

    # add avr for each measure
    dat = np.hstack((dat,dat.mean(1)[:,None]))
    best_score_ = dat[0,-1]

    columns = [str(i) for i in np.arange(idx.shape[0])]+['avr.']
    index = [i[1:].ljust(4)+'|' for i in index]
    tab = pd.DataFrame(np.abs(dat),index=index, columns = columns)

    if verbose:print(tab)

    return {
            'best_score': best_score_,
            'best_params' : best_params,
            'table': dat,
            'index': index,
            'columns': columns,
            'file_paths': paths_for_folds[str(best_params)],
            }
