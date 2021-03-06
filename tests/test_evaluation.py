import os
import h5py 
import numpy as np

from sklarge import GridSearchCV, run_local, evaluation, print_summary 
from sklearn.linear_model import SGDRegressor, SGDClassifier 
from sklearn.multioutput import MultiOutputClassifier , MultiOutputRegressor
import glob

pwd = os.path.dirname(os.path.abspath(__file__))


X = [pwd+'/data/X000000.h5', pwd+'/data/X000001.h5']
y = [pwd+'/data/y000000.h5', pwd+'/data/y000001.h5']
idx = [ [[0],[1]], [[1],[0]] ]

clf = MultiOutputClassifier(SGDClassifier())
reg = MultiOutputRegressor(SGDRegressor())
para = {
        'estimator__alpha': 10.**np.arange(-2,3),
        'estimator__penalty': ['l1', 'l2'],
        }


class testcase:

    def test_evaluation_basic(self):

        GS = GridSearchCV(
                estimator =  clf,
                param_grid = para,
                verbose=1
                )
        GS._create_job_files(X,y,idx,pwd+'/tmp_eval',mode='w')
        run_local(pwd+'/tmp_eval',-1)
        out = evaluation(pwd+'/tmp_eval')
        assert(out['table'].shape==(5,11))
        assert(np.max(out['table'])>0)
        assert(np.min(out['table'])<0)

        out = evaluation(pwd+'/tmp_eval',condition={'estimator__penalty':'l2'})
        assert(out['best_params']['estimator__penalty']=='l2')
        assert(out['table'].shape==(5,11))
        assert(np.max(out['table'])>0)
        assert(np.min(out['table'])<0)

        out = evaluation(pwd+'/tmp_eval',condition={'estimator__penalty':'l1'})
        assert(out['best_params']['estimator__penalty']=='l1')
        assert(out['table'].shape==(5,11))
        assert(np.max(out['table'])>0)
        assert(np.min(out['table'])<0)

    def test_merge_to_excel(self):

        res = []
        exp = []
        for estimator,name in zip([clf,reg],['clf','reg']):
            GS = GridSearchCV(
                    estimator =  estimator,
                    param_grid = para,
                    verbose=1
                    )
            out = pwd+'/tmp_eval_'+name
            GS._create_job_files(X,y,idx,out,mode='w')
            run_local(out,-1)
            res.append(evaluation(out,verbose=1))
        out = print_summary(res,['clf','reg'],save_excel='tmp.xlsx')
        print(len(list(out.keys()))==5)


if __name__ == "__main__":
    import nose
    nose.run(defaultTest=__file__, env={'NOSE_NOCAPTURE' : 1})
