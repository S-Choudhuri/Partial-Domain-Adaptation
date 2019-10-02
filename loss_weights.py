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
    for i in range(ysb.shape[0]):
        w_sample_class[i] = w_class[ysbi[i]]
        w_sample_adv[i] = w_class[ysbi[i]]

    return w_sample_class, w_sample_adv