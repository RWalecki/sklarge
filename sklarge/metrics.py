import numpy as np
from sklearn.metrics import make_scorer

def _pre_process(y_hat, y_lab):
    y_hat = np.array(y_hat, dtype=np.float64).T
    y_lab = np.array(y_lab, dtype=np.float64).T

    assert np.all(y_hat.shape == y_lab.shape)
    if len(y_hat.shape) == 1:
        y_hat = np.expand_dims(y_hat, axis=0)
        y_lab = np.expand_dims(y_lab, axis=0)
    return y_hat, y_lab

def _post_process(res):
    res[np.isnan(res)] = 0
    if len(res) == 1:
        return res[0]
    return res

def acc(y_hat, y_lab):
    y_hat, y_lab = _pre_process(y_hat, y_lab)
    return _post_process(np.mean((y_hat==y_lab), 1))
acc = make_scorer(acc,greater_is_better=True)

def mae(y_hat, y_lab):
    y_hat, y_lab = _pre_process(y_hat, y_lab)
    return _post_process(np.mean(np.abs(y_hat-y_lab), 1))
mae = make_scorer(mae,greater_is_better=False)

def mse(y_hat, y_lab):
    y_hat, y_lab = _pre_process(y_hat, y_lab)
    return _post_process(np.mean((y_hat-y_lab)**2, 1))
mse = make_scorer(mse,greater_is_better=False)

def rmse(y_hat, y_lab):
    y_hat, y_lab = _pre_process(y_hat, y_lab)
    return _post_process(np.mean((y_hat-y_lab)**2, 1)**0.5)
rmse = make_scorer(rmse,greater_is_better=False)

def icc(y_hat, y_lab, cas=3, typ=1):
    y_hat, y_lab = _pre_process(y_hat, y_lab)

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

    res[np.isnan(res)] = 0
    return _post_process(res.astype('float32'))
icc = make_scorer(icc,greater_is_better=True)

def pcc(y_hat, y_lab):
    y_hat, y_lab = _pre_process(y_hat, y_lab)
    res = []
    for y1, y2 in zip(y_lab, y_hat):
        res.append(np.corrcoef(y1, y2)[0, 1])
    res = np.array(res)
    res[np.isnan(res)] = 0
    return _post_process(res)
pcc = make_scorer(pcc,greater_is_better=True)

def f1_det(y_hat, y_lab):
    y_hat, y_lab = _pre_process(y_hat, y_lab)
    y_hat = y_hat>0
    y_lab = y_lab>0

    TP = np.sum((y_lab == 1)*(y_hat == 1), 1)
    FP = np.sum((y_lab != 1)*(y_hat == 1), 1)
    FN = np.sum((y_lab == 1)*(y_hat != 1), 1)
    F1 = (2.*TP)/(2*TP+FP+FN).astype(float)

    F1[np.isinf(F1)] = 0
    return _post_process(F1)
f1_det = make_scorer(f1_det,greater_is_better=False)
