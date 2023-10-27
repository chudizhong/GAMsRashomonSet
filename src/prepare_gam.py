import numpy as np
import pandas as pd
import sys
import src.utils as utils
from sklearn.linear_model import LogisticRegression
import time
import pickle

import rpy2
from rpy2.robjects.packages import importr
from rpy2 import robjects
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()


def fit_fastsparse(X, y, tmp_lambda0=None, tmp_lambda2=None):
    """
    X, y: numpy arrays. X shape is n*(p+1) and y is either 1 or -1. 
    """
    base = importr('base')
    FastSparse = importr('FastSparse')
    np.random.seed(seed=3337)

    if tmp_lambda0 is None:  
        tmp_lambda0 = 1
    
    if tmp_lambda2 is None:
        tmp_lambda2 = 1e-5

    d = {'package.dependencies': 'package_dot_dependencies', 'package_dependencies': 'package_uscore_dependencies'}
    FastSparse = importr('FastSparse', robject_translations = d)

    fit = FastSparse.FastSparse_fit(X, y, loss="Logistic", penalty="L0L2", intercept=True, algorithm="CDPSI", maxSuppSize = 300, autoLambda=False, lambdaGrid=[tmp_lambda0], nGamma = 1, gammaMin = tmp_lambda2, gammaMax = tmp_lambda2)
   
    betas_fastSparse = base.as_matrix(FastSparse.coef_FastSparse(fit))
    betas_fastSparse = np.asarray(betas_fastSparse)


    return betas_fastSparse


def get_fastsparse(data, lamb0, lamb2):
    X_orig, counts = utils.one_hot_encoding(data.iloc[:,:-1], one_hot=False) # n*p, no intercept column
    y_orig = pd.DataFrame(data.iloc[:,-1]) # {0,1}
    header = list(X_orig.columns)
    header = pd.Index(["intercept"] + header)
    header = header.astype("object")
    print("header dimension", len(header), flush=True)

    X, y = utils.get_X_y(X_orig, y_orig) # add a column of one to X_orig and make y in {1,-1}


    # Important to reweight the lamb0 before feed into the fastsparse algorithm
    w = fit_fastsparse(X_orig.values, y, tmp_lambda0=lamb0*y.shape[0], tmp_lambda2=lamb2)
    w = w.ravel() # (p+1, ) 
    acc, auc = utils.get_acc_and_auc(w, X, y)
    print("lamb0:{}, lamb2:{}, acc:{}, auc:{}, supp_size:{}".format(lamb0, lamb2, acc, auc, np.count_nonzero(w)), flush=True)
    
    return w, y, header

def prepare_sparse_gam(dname, lamb0, lamb2, multiplier):
    data = pd.read_csv("datasets/{}.csv".format(dname))

    lamb = 2 * lamb2

    w, y, header = get_fastsparse(data, lamb0, lamb2)

    y = y.ravel()
    X_new, header_new = utils.binary_to_one_hot(data.iloc[:,:-1], w, header)
    sample_p = X_new.sum(0)/X_new.shape[0]
    # sample_p[0] = 1e-5
    assert(sample_p.min()!=0)
    X_new_normalized = X_new/np.sqrt(sample_p)
    LR_model = LogisticRegression(penalty="l2", C=1/lamb, fit_intercept=True, solver='liblinear', intercept_scaling=10000, max_iter=1000)
    LR_model.fit(X_new_normalized[:,1:], (y+1)//2) # change y to {0,1}
    print(LR_model.intercept_.shape)
    print(LR_model.coef_.shape)
    w_new_normalized = np.c_[LR_model.intercept_, LR_model.coef_]
    w_new = w_new_normalized/np.sqrt(sample_p)
    w_new_normalized = w_new_normalized.ravel()
    w_new = w_new.ravel() # (m+1,) np array
    print(w_new)

    print(X_new.shape, w_new.shape, X_new_normalized.shape, w_new_normalized.shape)
    log_loss = utils.get_log_loss(X_new, y, w_new, tmp_lambda2, sample_p)
    log_loss_normalized = utils.get_log_loss(X_new_normalized, y, w_new_normalized, tmp_lambda2, np.ones(X_new.shape[1]))
    print('objective:', log_loss, "objective in LR", log_loss_normalized)

    H = utils.hessian(w_new, X_new, y, lamb2, sample_p)


    outfile = "{}_{}_{}_{}.p".format(dname, lamb0, lamb2, multiplier)
    eps = log_loss * multiplier
    print("m:{}, log objective:{}, eps:{}".format(multiplier, log_loss, eps))

    
    results = {
        "date": time.strftime("%d/%m/%y", time.localtime()),
        "data_file": dname,
        "X": X_new,
        "header_new": header_new,
        "p": w_new.shape[0], # including intercept, (m+1,)
        "sample_proportion": sample_p, 
        "lamb0": tmp_lambda0,
        "lamb2": tmp_lambda2,
        "multiplier": multiplier, 
        "rset_bound": eps, 
        "w_orig": w_new, 
        "log_loss_orig": log_loss,
        "hessian": H
    }
    
    with open(outfile, 'wb') as out:
        pickle.dump(results, out, protocol=pickle.DEFAULT_PROTOCOL)
    
    return outfile
