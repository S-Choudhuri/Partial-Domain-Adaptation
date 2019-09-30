import numpy as np

def weight_comp(y):
	cw = np.array(([0] * y.shape[1]))
	cn = np.array(([0] * y.shape[1]))
    yi = y.argmax(1)

    for i in range(ys.shape[0]):
        cw[yi[i]] = cw[yi[i]] + y[i, yi[i]]
        cn[yi[i]] = cn[yi[i]] + y[i, yi[i]]
 
    return cw / yi

def ingegrated_loss_weight(ws, wt):
	return 1 - np.absolute(ws-wt)

def loss_weight(ys, yt):
	ws = weight_comp(ys)
	wt = weight_comp(yt)	
    return integrated_loss_weight(ws, wt)
