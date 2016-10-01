import glob
import pandas as pd
from collections import defaultdict
import dill
import numpy as np

pd.set_option('expand_frame_repr', False)

def evaluation(path, verbose = 1, best_joint=True):
    if verbose:
        print('jobs:',len(glob.glob(path+'/*/setting.dlz')))
        print('done:',len(glob.glob(path+'/*/results.csv')))
    avr_res = defaultdict(list)
    file_paths = defaultdict(list)
    uniq_para = {}
    done_experiments = glob.glob(path+'/*/results.csv')
    if len(done_experiments)==0:
        print('could not find any results in \n',path)
        return 0, 0, 0, 0

    for f in glob.glob(path+'/*/results.csv'):
        dat = dill.load(open(f.rsplit('/',1)[0]+'/setting.dlz','rb'))
        res = np.genfromtxt(f.rsplit('/',1)[0]+'/results.csv',dtype=str,delimiter=',')
        if len(res.shape)==1:res = res[None,:]

        index = res[:,0]
        res = np.float16(res[:,1:])
        avr_res[str(dat['para'])].append(res)
        file_paths[str(dat['para'])].append(f)
        uniq_para[str(dat['para'])]=dat['para']

    # get tables with results
    paras, table = [],[]
    for para in avr_res:
        avr_res[para] = np.mean(np.stack(avr_res[para]),0)
        table.append(avr_res[para])
        paras.append(para)


    table = np.stack(table).transpose(1,0,2)

    # results of first metric in table [para X output]
    if best_joint==True:
        idx = np.tile(np.argmax(table[-1].mean(1)),table.shape[2])
    else:
        idx = np.argmax(table[-1],0)

    best_params_ = uniq_para[paras[np.unique(idx)[0]]]

    # store all results in one table [measures X output]
    dat = np.vstack([tab_[idx,np.arange(idx.shape[0])] for tab_ in table])

    # add avr for each measure
    dat = np.hstack((dat,dat.mean(1)[:,None]))
    best_score_ = dat[0,-1]

    columns = [str(i) for i in np.arange(idx.shape[0])]+['avr.']
    tab = pd.DataFrame(np.abs(dat),index=index, columns = columns)

    pd.set_option('display.float_format', lambda x: '%.2f' % x)
    if verbose:print(tab)
    return best_score_, best_params_, table, file_paths[str(best_params_)]
