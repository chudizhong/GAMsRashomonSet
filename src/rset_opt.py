import numpy as np
import pandas as pd
import pickle
import src.utils as utils
import time
import warnings
import torch
import torch.nn as nn
from itertools import combinations

class RSetOPT:
    def __init__(self, filepath, C=500, lr=0.0001):
        with open(filepath, "rb") as f:
            out = pickle.load(f)
        self.filepath = filepath
        self.dname = out["data_file"]
        data = pd.read_csv("datasets/{}.csv".format(self.dname))
        y = data.iloc[:,-1].values
        if np.min(y) == 0:
            y_max, y_min = np.max(y), np.min(y)
            y = -1 + 2 * (y-y_min)/(y_max-y_min)
        y = y.ravel()
        self.y = y
        self.X = out["X"]
        self.sample_p = out["sample_proportion"]
        self.N = self.X.shape[0]
        self.P = out["p"]
        self.xlabel = utils.get_xlabel(data, out["header_new"])
        self.w_orig = out["w_orig"]
        self.H = out["hessian"]
        self.lamb0 = out["lamb0"]
        self.lamb2 = out["lamb2"]
        self.multiplier = out["multiplier"]
        self.rset_bound = out["rset_bound"]
        self.ub = (self.rset_bound/self.multiplier) * (self.multiplier-1)
        self.C = C

        Sigma, V = np.linalg.eigh(self.H)
        self.H_half = torch.tensor((np.sqrt(Sigma) * V).T,requires_grad=True)
        self.w_center = torch.tensor(self.w_orig, requires_grad=True)
        self.optimizer = torch.optim.Adam([self.H_half, self.w_center], lr=lr)

    def get_normalized_H(self):
        """
        the ellipsoid defined by H_half is
        1/2 * (w-w_orig)^T H (w-w_orig) <= ub
        sometime we want to get rid of 1/2 and ub to make it more principled
        """
        return self.H / (2*self.ub)
    
        
    def sample_in_ellipsoid(self, n_samples):
        H, ub = self.H, self.ub
        d = self.P
        u = np.random.normal(size=(n_samples,d)) # randomly sample iid gaussian
        u = u/(np.linalg.norm(u,axis=1).reshape(-1,1)) # normalize to get uniformly random unit vectors
        r = (np.random.random(size=n_samples))**(1/d) # sample radius (uniformly in a sphere)
        # print(r)
        x_ = u * r.reshape(-1,1) # x_ is a uniformly random point in a sphere
        
        lamb, V = np.linalg.eigh(H) # eigen decomposition
        a = np.sqrt(2*ub/lamb) # scaling factor
        dw_samples = ((a*V) @ x_.T).T # transformation to a ellipsoid
        w_samples = dw_samples + self.w_orig

        return w_samples

    def get_precision(self):
        w_samples = self.sample_in_ellipsoid(10000)
        n_in_rset = 0
        for w_sample in w_samples:
            log_loss = utils.get_log_loss(self.X, self.y, w_sample, self.lamb2, self.sample_p)
            n_in_rset += int(log_loss<=self.rset_bound)
        precision = n_in_rset / w_samples.shape[0]
        print(precision,self.rset_bound)
        return precision

    def get_log_loss_torch(self, w_samples):
        X_torch = torch.tensor(self.X)
        y_torch = torch.tensor(self.y)
        sample_p = torch.tensor(self.sample_p)
        y_logit = w_samples @ X_torch.T
        losses = torch.log(1+torch.exp(-y_torch*y_logit))
        # print("avg log loss:{}, std log loss:{}".format(np.mean(loss), np.std(loss)))
        log_losses = losses.mean(1) + self.lamb2 * (sample_p[1:] * (w_samples[:,1:]**2)).sum(1)
        return log_losses

    def total_loss(self, w_samples):
        loss_det = self.H_half.det().abs() ** (1/self.P)
        losses_log = self.get_log_loss_torch(w_samples)
        loss_outrset = torch.clamp(losses_log-self.rset_bound,0).mean()
        # print('precision = ', (losses_log<=self.rset_bound).float().mean())

        return loss_det + self.C * loss_outrset
    
    def sample_in_ellipsoid_torch(self, n_samples=256):
        d = self.H.shape[0]
        u = torch.normal(0,1, size=(n_samples,d)) # randomly sample iid gaussian
        u = u/(u.norm(dim=1).reshape(-1,1)) # normalize to get uniformly random unit vectors
        r = (torch.rand(size=(n_samples,)))**(1/d) # sample radius (uniformly in a sphere)
        x_ = u * r.reshape(-1,1).double() # x_ is a uniformly random point in a sphere

        loss_w_center = self.get_log_loss_torch(self.w_center.reshape(1,-1))
        ub = self.ub
        dw_samples = ( self.H_half.inverse() @ x_.t() * ((ub*2)**0.5) ).t() # transformation to a ellipsoid
        w_samples = dw_samples + self.w_center
        
        return w_samples

    def finetune_ellipsoid(self, n_iters = 1000):
        print('----------- before optimization -----------')
        print('volume proportional to ', 1/self.H_half.det().abs())
        for i in range(n_iters):
            self.optimizer.zero_grad()
            w_samples = self.sample_in_ellipsoid_torch()
            loss = self.total_loss(w_samples)
            print(i, loss, flush=True)
            loss.backward()
            self.optimizer.step()

        print('----------- after optimization -----------')
        print('volume proportional to ', 1/self.H_half.det().abs())
        self.H = (self.H_half.T @ self.H_half).detach().numpy()
        self.w_orig = self.w_center.detach().numpy()

    def update_file(self, H_new, w_new):
        with open(self.filepath, "rb") as f:
            res = pickle.load(f)
        res["H_opt"] = H_new
        res["w_opt"] = w_new
        with open(self.filepath, 'wb') as f:
            pickle.dump(res, f, protocol=pickle.DEFAULT_PROTOCOL)
    
    