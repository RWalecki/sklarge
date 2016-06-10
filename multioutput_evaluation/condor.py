from metrics import pcc, icc, mse, f1_detection 
from sklearn.model_selection import ParameterGrid

def run(clf, X, y, S, folds=5, metric=pcc, n_jobs=-1, independent=True, out='/tmp/', mode='w', verbose=1):
    for i in ParameterGrid(clf.parameter):print i
    pass

def _run(clf, X, y, S, folds=5, metric=pcc, n_jobs=-1, independent=True, out='/tmp/', mode='w', verbose=1):
    if n_jobs==-1:n_jobs=multiprocessing.cpu_count()
    if out[-1]!='/':out+='/'
    folds = min([folds,len(np.unique(S))])

    name = clf.__class__.__name__
    '''
    make meta file
    '''
    if mode=='w':shutil.rmtree(out, ignore_errors=True)
    if not os.path.exists(out):os.makedirs(out)
    _create_jobs( clf, X, y, S, folds, out, mode)

def _create_jobs( clf, X, y, S, folds, out ,mode):
    cv = model_selection.LabelKFold(folds)

    jobs = []
    for fold in cv.split(X,y,S):
        for params in ParameterGrid(clf.parameter):
            jobs.append([params,fold,clf.estimator])

    for i,[fold,params,estimator] in enumerate(jobs):
        pwd_out = out+'/'+str(i).zfill(6)
        if not os.path.exists(pwd_out):os.makedirs(pwd_out)
        np.savez_compressed(pwd_out+'/setup',fold==fold, params=params, estimator=estimator)


def _create_jobs( clf, X, y, S, cv, out ,mode):
    for i, [fold, para] in enumerate(itertools.product(cv.split(S,S,S),ParameterGrid(clf.parameter))):
        pwd_out = out+str(i).zfill(6)
        if not os.path.exists(pwd_out):os.makedirs(pwd_out)
        setup = {}
        setup['fold']=fold
        setup['para']=para
        setup['X'] = X
        setup['y'] = y
        setup['estimator']=clf.estimator
        cPickle.dump(setup,gzip.open(pwd_out+'/setup.pklz','wb'))

def _run_jobs_local(out):
    args = glob.glob(out+'/*/*.pklz')
    _eval(args[0])

def _eval(pwd):
    setup = cPickle.load(gzip.open(pwd,'rb'))
    pwd_data = setup['X'].rsplit('/',1)[0]
    f = h5py.File(pwd_data)
    i = setup['fold'][0]
    X_tr = f[str(setup['X']).rsplit('/',1)[-1]][setup['fold'][0],:]
    X_te = f[str(setup['X']).rsplit('/',1)[-1]][setup['fold'][1],:]
    y_tr = f[str(setup['y']).rsplit('/',1)[-1]][setup['fold'][0],:]
    setup['estimator'].fit(X_tr,y_tr)
    out = {}
    out['y_hat'] = setup['estimator'].predict(X_te)
    out['y_lab'] = f[str(setup['y']).rsplit('/',1)[-1]][setup['fold'][1],:]
    cPickle.dump(out,gzip.open(pwd.rsplit('/',1)[0]+'/results.pklz','wb'))
