

import matplotlib.pyplot as plt
import numpy as np
import random
from ef import es

def multinorm_translate(_mu,_sigma):
    dim = _sigma.shape[0]
    #compute _sigma^(-1/2)
    eigva, eigvc = np.linalg.eig(_sigma)
    eigvc = np.matrix(eigvc)
    eigva_half = np.diag(eigva**0.5)
    sigma_half = eigvc.T * eigva_half * eigvc

    #normalized gaussion
    point = [random.gauss(0,1) for i in range(dim)]
    #transform
    point = point * sigma_half + _mu

    return np.array(point)[0].tolist()


def quadprog_equ(H, x, f, A, b):
        _f = np.array((H * np.matrix(x).T).T)[0] + f
        b = np.zeros(A.shape[0])
        up = np.column_stack((H,A.T))
        down = np.column_stack((A,np.zeros((A.shape[0],A.shape[0]))))
        X = np.vstack((up,down))

        b = np.append(-_f,b)
        result = np.array((np.linalg.inv(X) * np.matrix(b).T).T)[0]
        newx = result[:A.shape[1]]
        lamdas = result[A.shape[1]:]
        return newx, lamdas

def quadprog(x, H, f, A, b, Aeq, beq):
    RemoveSet ={}
    ActiveSet ={}
    for i in range(len(b)):
        if ((A[:,i].T*np.matrix(x).T)[0,0] - b[i])<0:
            ActiveSet[str(i)] = [np.matrix(A[:,i]).T, np.array([b[i]])]
        else:
            RemoveSet[str(i)] = [np.matrix(A[:,i]).T, np.array([b[i]])]

    maxiter = 200
    for i in range(200):
	if i % 10 ==0:
            print "iteration : ", i
        Ai = Aeq
        bi = beq

        for item in ActiveSet:
            Ai = np.column_stack((Ai, ActiveSet[item][0]))
            bi = np.append(bi, ActiveSet[item][1])

        Ai = Ai.T

        [__x, lamda_vec] = quadprog_equ(H, x, f, A, b)

        lamda_vec=lamda_vec[Aeq.shape[1]:]
        if np.linalg.norm(__x) < 1e-4:
            lamda = np.min(lamda_vec)

            if lamda <= -1e-4:
                lamda_idx = np.where(lamda_vec==lamda)[0][0]
                if lamda_idx < len(ActiveSet.keys()):
                    constrainIdx = ActiveSet.keys()[lamda_idx]
                    RemoveSet[str(constrainIdx)] = ActiveSet[str(constrainIdx)]
                    del(ActiveSet[str(constrainIdx)])
            else:
                return x
        else:
            alpha = 1e10
            renew_constrain = -1
            for item in RemoveSet:
                [tmpA,tmpb] = RemoveSet[item]
                if (tmpA.T*np.matrix(__x).T)[0,0] > 0:
                    alphaTmp = (tmpb[0] - (tmpA.T*np.matrix(x).T)[0,0] )*1.0/(tmpA.T*np.matrix(__x).T)[0,0]
                    if alphaTmp < alpha:
                        alpha = alphaTmp
                        renew_constrain = item

            alpha = min(alpha,1)
            x = x + alpha * __x
            if alpha == 1:
                pass
            elif renew_constrain != -1:
                ActiveSet[renew_constrain] = RemoveSet[renew_constrain]    #xu yao tong shi wei hu yi ge hubu zidian
                del(RemoveSet[renew_constrain])

    return x


def svm(X,Y):
	cols = X.shape[1]

	X_ = X
	for i in range(cols):
		X_[:i] = Y[i]*X_[:i]
	X_ = np.matrix(X_)
	H = X_.T * X_
	f = -np.ones(cols)
	Aeq = np.matrix(Y).T
	A = -np.eye(cols)
	beq = np.array([0])
	b = np.zeros(cols)
	x_init = []
	pcnt = 0

        yy = np.array(Y)
	pcnt = len(np.where(yy==1)[0])
	ncnt = len(np.where(yy!=1)[0])

	for item in Y:
		if item == 1:
			x_init.append(1.0/pcnt)
		else:
			x_init.append(1.0/ncnt)
	x_init = np.array(x_init)

        alpha = quadprog(x_init, H , f, A, b, Aeq, beq)

	w = np.zeros(X.shape[0])
	for i in range(cols):
		w += alpha[i]*Y[i]*(np.array(X[:,i].T)[0])

	b1 = []
	b2 = []
	for i in range(cols):
		if Y[i] > 0:
			b1.append(Y[i]-(np.matrix(w)*np.matrix(X[:,i]).T)[0,0])
		else:
			b2.append(Y[i]-(np.matrix(w)*np.matrix(X[:,i]).T)[0,0])

	b = (sum(b1)+sum(b2))*1.0/cols
	
	wn = np.linalg.norm(w)
	w = w*1.0/wn
	b = b*1.0/wn
	return w,b





if __name__=='__main__':

	mu1 = np.array([0,-4])
	sigma1 = np.array([[3,0],[0,4]])
    
        pdata = np.zeros((0,2))
	pn = 400
	for i in range(pn):
		point =  multinorm_translate(mu1, sigma1)
                pdata = np.vstack((pdata,point))
        
        print pdata.shape
		
	plabel = np.ones(pn)

	mu2 = np.array([1,3])
	sigma2 = np.array([[2,0],[0,3]])
	ndata = np.zeros((0,2))
	nn = 400
	for i in range(nn):
		point =  multinorm_translate(mu2, sigma2)
		ndata= np.vstack((ndata,point))

	nx = []
	ny = []
	for item in ndata:
		nx.append(float(item[0]))
		ny.append(float(item[1]))

	
	nlabel = -np.ones(nn)

	X = np.column_stack((pdata.T,ndata.T))
        print X.shape
	Y = np.append(plabel,nlabel)
	[w,b] = svm(X,Y) 

        px = []
	py = []
	for item in pdata:
		px.append(float(item[0]))
		py.append(float(item[1]))
	nx = []
	ny = []
	for item in ndata:
		nx.append(float(item[0]))
		ny.append(float(item[1]))
		
	plt_x = np.arange(-6, 6, 0.3)
	plt_y = (-b-w[0]*plt_x)*1.0/w[1]
	plt.plot(plt_x, plt_y, '-')
	plt.scatter(px, py , 10, color ='blue', marker = 'o')
	plt.scatter(nx, ny , 10, color ='red', marker = 'o')
	plt.show()
