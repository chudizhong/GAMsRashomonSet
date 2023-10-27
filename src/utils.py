import numpy as np
import pandas as pd
import sklearn
import sys

def get_log_loss(X, y, w, lamb2, sample_p):
    """
    X: (n*p+1) array
    y: (n,) array {-1,1}
    w: (p+1,) array
    lamb2: lambda 2
    sample_p: (p+1,) array
    """
    y_logit = X @ w
    loss = np.log(1+np.exp(-y*y_logit))
    log_loss = np.mean(loss) + lamb2 * (sample_p[1:] * (w[1:]**2)).sum()
    return log_loss

def hessian(w, X, y, lamb2, sample_p):
    '''
    w: p+1 from fastsparse
    X: n*p+1 array
    y: n array, +1, -1
    lamb2: lambda 2
    sample_p: (p+1,) array
    '''
    n, d = X.shape[0], X.shape[1]
    prob = (1 / (1 + np.exp(-X @ (w.reshape(-1,1))))).ravel()
    D = np.diag(prob * (1-prob))
    H = X.T @ D @ X / n
    H[(np.arange(1,d),np.arange(1,d))] += 2*lamb2*sample_p[1:]
    return H


def get_X_y(X, y):
    # X, y are dataframes and X is n*p 
    X=X.to_numpy()
    X0 = np.ones((X.shape[0],1), dtype='int8')
    X = np.hstack((X0,X))
    n,p=X.shape
    y=y.to_numpy()
    y_max, y_min = np.max(y), np.min(y)
    y = -1 + 2 * (y-y_min)/(y_max-y_min) # convert y to -1 and 1
    return X, y


def compute_loss_from_scores_real(scores):
    loss = np.sum(np.log(1+np.exp(-scores)))
    return loss

def compute_loss_from_betas(betas, X, y):
    Xy = y*X
    scores = Xy.dot(betas)
    return compute_loss_from_scores_real(scores)

def compute_accuracy(betas, X, y):
    y_pred = X.dot(betas)
    y_pred = 2 * (y_pred > 0) - 1
    return np.sum(y_pred == y.ravel()) / y.shape[0]

def compute_auc(betas, X, y):
    y_pred = X.dot(betas)
    y_pred = 1/(1+np.exp(-y_pred))
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y, y_pred)
    auc = sklearn.metrics.auc(fpr, tpr)
    return auc

def get_acc_and_auc(betas, X, y):
    accuracy = compute_accuracy(betas, X, y)
    auc = compute_auc(betas, X, y)
    return accuracy, auc


def one_hot_encoding(X, one_hot=True):
    # transform the original real dataset into one-hot or binary dataframe. 
    colnames = X.columns
    X = X.values
    counts = [np.unique(X[:,i]).shape[0] for i in range(X.shape[1])]
    X_trans = np.zeros((X.shape[0], sum(counts)), dtype="int8")
    header = []
    cum_counts = np.cumsum(counts)

    for i in range(X.shape[1]):
        uni = np.unique(X[:,i])
        length = len(uni)

        for j in range(length):
            if one_hot:
                if j==0:
                    cname = "{}<={}".format(colnames[i], uni[j])
                    k = X[:,i] <= uni[j]
                else:
                    cname = "{}<{}<={}".format(uni[j-1], colnames[i], uni[j])
                    k = (X[:,i] > uni[j-1]) & (X[:,i] <= uni[j])
                
            else:
                cname = "{}<={}".format(colnames[i], uni[j])
                k = X[:, i] <= uni[j]
            
            if i == 0:
                X_trans[k,j] = 1
            else:
                X_trans[k,cum_counts[i-1]+j] = 1
            header.append(cname)
    X_trans = pd.DataFrame(X_trans, columns = header)

    return X_trans, counts

def binary_to_one_hot(X, w, header):
    '''
    X: (n, p) feature matrix. Type: data frame. Each column is a continuous or categorical varaible
    w: (p'+1,) weights of logistic regression. Type: array. w[0] is intercept
    header: (p'+1, ) column names. Type: array. header[0] is intercept
    '''

    idx = np.where(w != 0.0)[0]
    n = X.shape[0]

    # add intercept 
    header_new = [header[0]]

    if idx[0] == 0:
        idx = idx[1:]
    header = header[idx] # effective header
    header_tmp = np.array([i.split("<=") for i in header])
    pair = {} # feature and threshold pairs 
 
    for i in range(header_tmp.shape[0]):
        f = header_tmp[i,0] # feature
        t = float(header_tmp[i,1]) # threshold # check
        if f not in pair:
            pair[f] = [t]
        else:
            pair[f].append(t)

    v_count = [0]
    print("pair before removing", pair)
    for k, v in pair.copy().items():
        t_max = np.max(X[k])   # k in string
        if t_max != v[-1]:
            v.append(t_max)

        print("k, v", k, v)
        if len(v) > 1:
            v_count.append(len(v))
        else:
            pair.pop(k)
    print("pair after removing", pair)
    print("v_count:", v_count)
    v_cumsum = np.cumsum(v_count)
    print(v_cumsum) 
    X_new = np.zeros((n, sum(v_count)))
    for i, (k,v) in enumerate(pair.items()):
        for j in range(len(v)):
            if j == 0:
                row_idx = X[k] <= v[0]
                header_new.append("{}<={}".format(k, v[0]))
            else:
                row_idx = (X[k] > v[j-1]) & (X[k] <= v[j])
                header_new.append("{}<{}<={}".format(v[j-1], k, v[j]))
            print(k, j, v_cumsum[i]+j)
            X_new[row_idx,v_cumsum[i] + j] = 1
    for i in range(1,len(v_cumsum)):
        print(np.allclose(np.ones(n), X_new[:, v_cumsum[i-1]:v_cumsum[i]].sum(1)))
    
    X0 = np.ones((n,1))
    X_new = np.hstack((X0,X_new))

    return X_new, header_new

def get_xlabel(data, header):
    xlabel = {}
    for i, k in enumerate(header):
        if i == 0:
            xlabel["bias"] = [0,1]
            continue # skip the intercept

        if "0"<= k[0] <= "9" or k[0]=='-':
            left_idx = k.find("<")
            right_idx = k.rfind("<")
            right = k[right_idx+2:]
            f = k[left_idx+1:right_idx]
            xlabel[f].append(float(right))
        else:
            f = k.split("<=")[0]
            min_val = data[f].min()
            xlabel[f] = [min_val, float(k.split("<=")[1])]
    return xlabel
