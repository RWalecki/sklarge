import multioutput_evaluation as me
from sklearn import model_selection
import numpy as np
import h5py 

f = h5py.File('./tests/data/test.h5')

CLF = [
        me.sk_estimator.MTL(),
        me.sk_estimator.SVR(),
        me.sk_estimator.SVC(),
        # me.tf_estimator.DNN_C(max_iter=1,verbose=-1),
        # me.tf_estimator.DNN_R(max_iter=1,verbose=-1),
        ]



class testcase:

    def test_gird_numpy(self):
        '''
        to do!
        '''
        pass

    def test_grid_hdf5(self):
        '''
        '''
        clf = me.sk_estimator.MVR()
        cv = model_selection.LabelKFold(2)
        clf = me.GridSearchCV(
                clf,
                clf.param_grid,
                cv=cv,
                n_jobs=-1,
                output = '/tmp/test_grid_hdf5',
                )
        clf.fit(
                X = './tests/data/test.h5/X',
                y = './tests/data/test.h5/y',
                labels = './tests/data/test.h5/S',
                )


if __name__ == "__main__":
    import nose
    nose.run(defaultTest=__file__, env={'NOSE_NOCAPTURE' : 1})
