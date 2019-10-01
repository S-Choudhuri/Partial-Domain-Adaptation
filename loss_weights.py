import numpy as np

def weight_comp(y):
    cw = np.array(([0.] * y.shape[1]))
    cn = np.array(([0.] * y.shape[1]))
    w = np.array(([0.] * y.shape[1]))
    yi = y.argmax(1)

    for i in range(y.shape[0]):
        cw[yi[i]] = cw[yi[i]] + y[i, yi[i]]
        cn[yi[i]] = cn[yi[i]] + 1
    for i in range(cw.shape[0]):
        if cn[i] == 0:
            w[i] = cn[i]
        else:
            w[i] = cw[yi[i]] / cn[yi[i]]
    return w

def integrated_loss_weight(ws, wt):
    return 1 - 20*np.absolute(ws-wt)

def loss_weight(ys, yt, ysb):
    ws = weight_comp(ys)
    wt = weight_comp(yt)
    w_class = integrated_loss_weight(ws, wt)
    w_sample_class = np.array(([0.] * 2 * ysb.shape[0]))
    w_sample_adv = np.array(([1.] * 2 * ysb.shape[0]))

    ysbi = ysb.argmax(1)
    for i in range(ysb.shape[0]):
        w_sample_class[i] = w_class[ysbi[i]]
        w_sample_adv[i] = w_class[ysbi[i]]

    return w_sample_class, w_sample_adv




