import os
import h5py 
import numpy as np

from sklarge import GridSearchCV, run_local, evaluation
from sklearn.linear_model import SGDRegressor 
from sklearn.multioutput import MultiOutputClassifier 
from sklarge.metrics import mse, pcc
import glob

pwd = os.path.dirname(os.path.abspath(__file__))


X = [pwd+'/data/X000000.h5', pwd+'/data/X000001.h5']
y = [pwd+'/data/y000000.h5', pwd+'/data/y000001.h5']

clf = MultiOutputClassifier(SGDRegressor())
para = {'estimator__alpha': 10.**np.arange(-2,3)}


class testcase:

    def test_cross_validation(self):

        idx = [ [[0],[1]], [[1],[0]] ]

        GS = GridSearchCV(
                estimator =  clf,
                param_grid = para,
                verbose=1
                )

        GS._create_job_files(X,y,idx,pwd+'/tmp',mode='w')
        run_local(pwd+'/tmp',-1)
        evaluation(pwd+'/tmp')

    def test_tr_te_split(self):

        idx = [ [ [0], [1] ] ]

        GS = GridSearchCV(
                estimator =  clf,
                param_grid = para,
                verbose=1
                )

        GS._create_job_files(X,y,idx,pwd+'/tmp',mode='w')
        run_local(pwd+'/tmp',-1)
        evaluation(pwd+'/tmp')

    def test_tr_tr_user_scoring(self):

        idx = [ [ [0], [0] ] ]

        GS = GridSearchCV(
                estimator =  clf,
                param_grid = para,
                scoring = [mse,pcc],
                verbose=1
                )

        GS._create_job_files(X,y,idx,pwd+'/tmp',mode='w')
        run_local(pwd+'/tmp',-1)
        evaluation(pwd+'/tmp')

    def test_continue_cross_validation(self):

        idx = [ [[0],[1]], [[1],[0]] ]

        GS = GridSearchCV(
                estimator =  clf,
                param_grid = para,
                verbose=1
                )

        GS._create_job_files(X,y,idx,pwd+'/tmp',mode='w')
        run_local(pwd+'/tmp',-1)
        evaluation(pwd+'/tmp')

        # remove 5 results 
        for done_job in glob.glob(pwd+'/tmp/*/results.csv')[:5]:
            os.remove(done_job)

        # compute missing results
        run_local(pwd+'/tmp',-1)
        evaluation(pwd+'/tmp')

if __name__ == "__main__":
    import nose
    nose.run(defaultTest=__file__, env={'NOSE_NOCAPTURE' : 1})
