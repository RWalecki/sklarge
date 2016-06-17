import multioutput_evaluation as me
from sklearn import model_selection
import numpy as np
import h5py 
import os
pwd = os.path.dirname(os.path.abspath(__file__))


f = h5py.File('./tests/data/test.h5')

CLF = [
        me.sk_estimator.MTL(),
        me.sk_estimator.SVR(),
        me.sk_estimator.SVC(),
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
        GS = me.GridSearchCV(
                clf,
                cv=cv,
                n_jobs=-1,
                )
        GS.fit(
                X = pwd+'/data/test.h5/X',
                y = pwd+'/data/test.h5/y',
                labels = pwd+'/data/test.h5/S',
                tmp = '/tmp/GridSearchCV',
                submit='local',
                )


if __name__ == "__main__":
    import nose
    nose.run(defaultTest=__file__, env={'NOSE_NOCAPTURE' : 1})
