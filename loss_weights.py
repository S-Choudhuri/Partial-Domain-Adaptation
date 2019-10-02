import numpy as np

def weight_comp(y):
    cw = np.sum(y, axis = 0)
    cw = cw / y.shape[0]
    return cw

def integrated_loss_weight(wt):
    w = wt / np.amax(wt)
    return w

def loss_weight(yt, ysb):
    wt = weight_comp(yt)
    w_class = integrated_loss_weight(wt)
    w_sample_class = np.array(([0.] * 2 * ysb.shape[0]))
    w_sample_adv = np.array(([1.] * 2 * ysb.shape[0]))

    ysbi = ysb.argmax(1)
    w_sample_class[0:ysb.shape[0]] = w_class[ysbi]
    w_sample_adv[0:ysb.shape[0]] = w_class[ysbi]

    return w_sample_class, w_sample_adv