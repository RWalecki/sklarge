import numpy as np
from sklearn.metrics import make_scorer

def _process(y_hat, y_lab, fun):
    '''
    - split y_true and y_pred in lists
    - removes frames where labels are unknown (-1)
    - returns list of predictions
    '''
    y1 = [x for x in y_hat.T]
    y2 = [x for x in y_lab.T]
    
    out = []
    for i, [_y1, _y2] in enumerate(zip(y1, y2)):
        idx = _y2!=-1
        _y1 = _y1[idx]
        _y2 = _y2[idx]
        if np.all(_y2==-1):
            out.append(np.nan)
        else:
            out.append(fun(_y1,_y2))
    return np.array(out)

def _acc(y_hat, y_lab):
    def fun(y_hat,y_lab):
        y_hat = np.int16(y_hat)
        y_lab = np.int16(y_lab)
        return np.mean(y_hat==y_lab)
    return _process(y_hat, y_lab, fun)
acc = make_scorer(_acc,greater_is_better=True)

def _mae(y_hat, y_lab):
    def fun(y_hat,y_lab):
        y_hat = np.int16(y_hat)
        y_lab = np.int16(y_lab)
        return np.mean(np.abs(y_hat-y_lab))
    return _process(y_hat, y_lab, fun)
mae = make_scorer(_mae,greater_is_better=False)

def _mse(y_hat, y_lab):
    def fun(y_hat,y_lab):
        y_hat = np.int16(y_hat)
        y_lab = np.int16(y_lab)
        return np.mean((y_hat-y_lab)**2)
    return _process(y_hat, y_lab, fun)
mse = make_scorer(_mse,greater_is_better=False)

def _rmse(y_hat, y_lab):
    def fun(y_hat,y_lab):
        y_hat = np.int16(y_hat)
        y_lab = np.int16(y_lab)
        return (np.mean((y_hat-y_lab)**2))**0.5
    return _process(y_hat, y_lab, fun)
rmse = make_scorer(_rmse,greater_is_better=False)

def _icc(y_hat, y_lab, cas=3, typ=1):
    def fun(y_hat,y_lab):
        y_hat = y_hat[None,:]
        y_lab = y_lab[None,:]

        Y = np.array((y_lab, y_hat))
        # number of targets
        n = Y.shape[2]

        # mean per target
        mpt = np.mean(Y, 0)

        # print mpt.eval()
        mpr = np.mean(Y, 2)

        # print mpr.eval()
        tm = np.mean(mpt, 1)

        # within target sum sqrs
        WSS = np.sum((Y[0]-mpt)**2 + (Y[1]-mpt)**2, 1)

        # within mean sqrs
        WMS = WSS/n

        # between rater sum sqrs
        RSS = np.sum((mpr - tm)**2, 0) * n

        # between rater mean sqrs
        RMS = RSS

        # between target sum sqrs
        TM = np.tile(tm, (y_hat.shape[1], 1)).T
        BSS = np.sum((mpt - TM)**2, 1) * 2

        # between targets mean squares
        BMS = BSS / (n - 1)

        # residual sum of squares
        ESS = WSS - RSS

        # residual mean sqrs
        EMS = ESS / (n - 1)

        if cas == 1:
            if typ == 1:
                res = (BMS - WMS) / (BMS + WMS)
            if typ == 2:
                res = (BMS - WMS) / BMS
        if cas == 2:
            if typ == 1:
                res = (BMS - EMS) / (BMS + EMS + 2 * (RMS - EMS) / n)
            if typ == 2:
                res = (BMS - EMS) / (BMS + (RMS - EMS) / n)
        if cas == 3:
            if typ == 1:
                res = (BMS - EMS) / (BMS + EMS)
            if typ == 2:
                res = (BMS - EMS) / BMS

        res = res[0]

        if np.isnan(res) or np.isinf(res):
            return 0
        else:
            return res

    return _process(y_hat, y_lab, fun)

icc = make_scorer(_icc,greater_is_better=True)

def _pcc(y_hat, y_lab):

    def fun(y1, y2):
        res = np.corrcoef(y1, y2)[0, 1]
        if np.isnan(res) or np.isinf(res):
            return 0
        else:
            return res
    return _process(y_hat, y_lab, fun)
pcc = make_scorer(_pcc,greater_is_better=True)



if __name__ == "__main__":
    import numpy as np
    y1 = np.random.randint(0,5,[100,3])
    y2 = np.random.randint(0,5,[100,3])
    y1[:,0] = y2[:,0]
    y2[:50,0]=-1
    y1[:50,0]=-1
    y2[:,-1]=-1

    print(_acc(y1,y2))
    print(_mae(y1,y2))
    print(_mse(y1,y2))
    print(_rmse(y1,y2))
    print(_icc(y1,y2))
    print(_pcc(y1,y2))
