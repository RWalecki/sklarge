import cPickle,dill, gzip
import glob
import h5py 
import os
import multiprocessing
def _run_job(path,mode='w'):
    output = pwd.rsplit('/',1)[0]+'/predictions.dlz'
    if mode=='c':
        if os.path.isfile(output):return 0
    exp = dill.load(gzip.open(path,'rb'))
    f_h5 = h5py.File(exp['pwd_X'].rsplit('/',1)[0])
    X = f_h5[exp['pwd_X'].rsplit('/',1)[1]]
    y = f_h5[exp['pwd_y'].rsplit('/',1)[1]]
    tr,te = exp['data_split']
    clf = exp['estimator']
    clf.fit(X,y,tr)
    y_hat = clf.predict(X,te)
    f_h5.close()

    cPickle.dump(y_hat,gzip.open(output,'wb'))



if __name__ == '__main__':
    # single process
    for i,f in enumerate(glob.glob('/tmp/bbb/*/setting.dlz')):
        _run_job(f)

    # multi processing
    p = multiprocessing.Pool(12)
    p.map(_run_job, glob.glob('/tmp/bbb/*/setting.dlz') )
