import multioutput_evaluation as me
import numpy as np
import h5py 

f = h5py.File('./tests/data/test.h5')

CLF = [
        me.sk_estimator.MVR(),
        me.sk_estimator.MTL(),
        me.sk_estimator.SVR(),
        me.sk_estimator.SVC(),
        # me.tf_estimator.DNN_C(max_iter=1,verbose=-1),
        # me.tf_estimator.DNN_R(max_iter=1,verbose=-1),
        ]



class testcase:

    def test_numpy(self):
        X = np.array(f['X'][::])
        y = np.array(f['y'][::])
        for clf in CLF:
            clf.fit(X,y)
            y_hat = clf.predict(X)
            assert y_hat.shape==(100,10)

    def test_hdf5(self):
        for clf in CLF:
            clf.fit(f['X'],f['y'])
            y_hat = clf.predict(f['X'])
            assert y_hat.shape==(100,10)

    def test_hdf5_split(self):
        for clf in CLF:
            clf.fit(f['X'],f['y'],np.arange(0,30))
            y_hat = clf.predict(f['X'],np.arange(50,70))
            assert y_hat.shape==(20,10)


if __name__ == "__main__":
    import nose
    nose.run(defaultTest=__file__, env={'NOSE_NOCAPTURE' : 1})
