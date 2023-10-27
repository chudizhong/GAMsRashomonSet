import sys
# sys.path.append('/usr/pkg/cplex-studio-221/cplex/python/3.10/x86-64_linux/')
# import cplex
# from cplex.exceptions import CplexError
import cvxpy as cp
import numpy as np
import pandas as pd
import pickle
import src.utils as utils
from matplotlib import pyplot as plt
import time
import warnings
import torch
import torch.nn as nn
from itertools import combinations
from math import comb
import random

class RSetGAMs:
    def __init__(self, filepath):
        with open(filepath, "rb") as f:
            out = pickle.load(f)
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
        self.w_init = out["w_orig"]
        self.H_init = out["hessian"]
        self.w_orig = out["w_opt"]
        self.H = out["H_opt"]
        self.lamb0 = out["lamb0"]
        self.lamb2 = out["lamb2"]
        self.multiplier = out["multiplier"]
        self.rset_bound = out["rset_bound"]
        self.ub = (self.rset_bound/self.multiplier) * (self.multiplier-1)


    def get_merge_ranges(self, n_support_set, max_n_ranges = 10000):
        xlabel = self.xlabel
        n_features = len(xlabel)
        n_bins_feature = [len(xlabel[f])-1 for f in xlabel]
        offsets = np.cumsum([0] + n_bins_feature)
        start_indices = []
        for i, n_bins in enumerate(n_bins_feature):
            start_indices += list(range(offsets[i]+1, offsets[i]+n_bins))
        n_choices = n_support_set - n_features
        merge_ranges = []

        use_random = comb(len(start_indices), n_choices)>max_n_ranges # use random combinations because # possible combinations is too large
        for indices in combinations(start_indices, n_choices):
            if use_random:
                indices = sorted(random.sample(start_indices, n_choices))
            else:
                indices = list(indices)
            bin_edges = []
            i, j = 0, 0
            
            while j!=len(offsets):
                if i==len(indices) or offsets[j]<indices[i]:
                    bin_edges.append(offsets[j])
                    j+=1
                else:
                    bin_edges.append(indices[i])
                    i+=1
            
            bin_index_ranges = [[bin_edges[i], bin_edges[i+1]-1] for i in range(len(bin_edges)-1) if bin_edges[i]<bin_edges[i+1]-1]
            merge_ranges.append(bin_index_ranges)

            if len(merge_ranges)>=max_n_ranges:
                break # break if exceeding the max # ranges
        return merge_ranges

    
    def merge_bins(self, H_orig, w_center_orig, ub_orig, bin_index_ranges):
        """
        This function calculates the new ellipsoid of RSet after merging bins in the shape function
        H_orig: the original H matrix in the ellipsoid
        w_center_orig: the original center of ellipsoid
        ub_orig: the original ub (rhs of the ellipsoid inequality)
        bin_index_ranges: list of lists, each item is [start_index, end_index], denoting the original feature from start_index to end_index will be merged
        """
        deleted_indices = set()
        H_orig = H_orig.copy()
        H_w_orig = H_orig @ w_center_orig.reshape(-1,1)
        w_H_w_orig = (w_center_orig.reshape(1,-1) @ H_orig @ w_center_orig.reshape(-1,1)).item(0)
        for start_index, end_index in bin_index_ranges:
            H_orig[start_index] += H_orig[start_index+1:end_index+1].sum(0)
            H_orig[start_index+1:end_index+1] = 0
            H_orig[:, start_index] += H_orig[:, start_index+1:end_index+1].sum(1)
            H_orig[:, start_index+1:end_index+1] = 0

            H_w_orig[start_index] +=  H_w_orig[start_index+1:end_index+1].sum()
            H_w_orig[start_index+1:end_index+1] = 0

            for index in range(start_index+1, end_index+1):
                deleted_indices.add(index)

        kept_indices = [i for i in range(H_orig.shape[0]) if i not in deleted_indices]
        H_new = H_orig[kept_indices][:,kept_indices]
        H_w_new = H_w_orig[kept_indices]
        w_center_new = (np.linalg.inv(H_new) @ H_w_new).ravel()
        w_H_w_new = (w_center_new.reshape(1,-1) @ H_new @ w_center_new.reshape(-1,1)).item(0)
        ub_new = ub_orig - 0.5*(w_H_w_orig - w_H_w_new)

        return H_new, w_center_new, ub_new
    
    def check_obj(self, w):
        log_loss = utils.get_log_loss(self.X, self.y, w, self.lamb2, self.sample_p)
        print("log obj:", log_loss, "rset_bound:", self.rset_bound)
        if log_loss > self.rset_bound:
            warnings.warn("solution is out of the Rset. ")
        return log_loss
   

    def get_f_idx(self, f):
        if f not in self.xlabel.keys():
            raise Exception("feature is invalid")
        l = 0
        for i, (k, v) in enumerate(self.xlabel.items()):
            if k == f:
                cnt = len(v)-1
                break
            else:
                l += len(v)-1
        return l, cnt
        
    def monotonicity(self, f, dir):
        """
        w_orig: (p+1,) array
        H: (p+1, p+1) array
        f: string, feature
        dir: string, ["increase", "decrease"]
        """
        l, cnt = self.get_f_idx(f)
        r = l + cnt
        w = cp.Variable(self.P)

        if dir == "increase":
            constraint = [w[j] >= w[j-1] for j in range(l+1, r)]
        elif dir == "decrease":
            constraint = [w[j] <= w[j-1] for j in range(l+1, r)]
        else:
            raise Exception("direction should be either increase or decrease")
        
        prob = cp.Problem(cp.Minimize(cp.quad_form(w-self.w_orig, self.H)),
                     constraint)

        prob.solve()
        print("Second solve time:", prob.solver_stats.solve_time)
        # print("\nThe optimal value is", prob.value)
        # print("A solution w is", w.value)

        w = w.value
        obj_w = self.check_obj(w) 
        return w


    def mcr_minus(self, f):
        l, cnt = self.get_f_idx(f)
    
        pr = np.r_[np.zeros(l), np.array([sum(self.X[:, l+c]/self.N) for c in range(cnt)])]
        pr = np.r_[pr, np.zeros(self.P - l - cnt)]
        pr = pr.reshape(self.P, 1)
        # print(pr)

        w = cp.Variable(self.P)  
        prob = cp.Problem(cp.Minimize(pr.T @ cp.abs(w)), 
                        [cp.quad_form(w-self.w_orig, self.H) <= 1] + [w[c] == self.w_orig[c] for c in range(l)] + 
                        [w[c] == self.w_orig[c] for c in range(cnt+l, self.P)]
                        )
        prob.solve()

        print("Second solve time:", prob.solver_stats.solve_time)
        # print("The optimal value is", prob.value)
        # print("A solution w is", np.round(w.value, 3))

        w_fix = w.value
        # print((w_fix - self.w_orig).T @ self.H @ (w_fix - self.w_orig))
        obj_w = self.check_obj(w_fix)

        w_all = cp.Variable(self.P)
        prob_all = cp.Problem(cp.Minimize(pr.T @ cp.abs(w_all)), 
                        [cp.quad_form(w_all-self.w_orig, self.H) <= 1]
                        )
        prob_all.solve()
        print("Second solve time:", prob_all.solver_stats.solve_time)
        # print("The optimal value is", prob_all.value)
        # print("A solution w is", np.round(w_all.value, 3))
        w_all = w_all.value

        # print((w_all-self.w_orig).T @ self.H @ (w_all-self.w_orig))
        obj_w_all = self.check_obj(w_all)
        return w_fix, prob.value, prob.solver_stats.solve_time, w_all, prob_all.value, prob_all.solver_stats.solve_time

    def mcr_plus_mip(self, f, fix=False):
        q_rhs = self.w_orig.reshape(1,self.P) @ self.H @ self.w_orig.reshape(self.P,1)
        q_rhs = 1 - q_rhs.item(0)
        Hw_orig = -2 * self.H @ self.w_orig.reshape(self.P,1)

        H = self.H.tolist()
        w_orig = list(self.w_orig)
        Hw_orig = Hw_orig.squeeze()
        Hw_orig = Hw_orig.tolist()
        # print(H)

        l, cnt = self.get_f_idx(f)
    
        pr = [sum(self.X[:, l+c]/self.N) for c in range(cnt)]
        print("pr", pr)
        M = 200
        coef_obj = pr + [0.0]*(2*cnt) + [0.0]*(self.P-cnt)
        if fix:
            var_ub = [100.0]*(2*cnt) + [1.0]*(cnt) + [w_orig[i] for i in range(l)] +[w_orig[l+cnt+i] for i in range(self.P-l-cnt)]
            var_lb = [0.0]*(cnt) + [-100.0]*(cnt) + [0.0]*(cnt) + [w_orig[i] for i in range(l)] +[w_orig[l+cnt+i] for i in range(self.P-l-cnt)]
        else:
            var_ub = [100.0]*(2*cnt) + [1.0]*(cnt) + [100.0]*(self.P-cnt)
            var_lb = [0.0]*(cnt) + [-100.0]*(cnt) + [0.0]*(cnt) + [-100.0]*(self.P-cnt)
        var_type = 'C'*(2*cnt) + 'I'*(cnt) + 'C'*(self.P-cnt)
        var_names = ["w_abs{}".format(i) for i in range(cnt)] + ["w{}".format(i) for i in range(cnt)] +\
                ["B{}".format(i) for i in range(cnt)] + ["w_other{}".format(i) for i in range(self.P-cnt)]
                            
        rhs = [0.0]*(cnt) + [-M]*cnt + [0.0]*(2*cnt) 
        sense = "G"*(2*cnt) + "L"*(2*cnt)

        cst = [[[i, cnt+i, 2*cnt+i], [-1.0, 1.0, M]] for i in range(cnt)] +\
            [[[i, cnt+i, 2*cnt+i], [-1.0, -1.0, -M]] for i in range(cnt)] +\
            [[[i, cnt+i], [-1.0, 1.0]] for i in range(cnt)] +\
            [[[i, cnt+i], [-1.0, -1.0]] for i in range(cnt)] 

        cst_names = ["bigM1_" + str(i) for i in range(cnt)] +\
                    ["bigM2_" + str(i) for i in range(cnt)] +\
                    ["cst_abs_w1_" + str(i) for i in range(cnt)] +\
                    ["cst_abs_w2_" + str(i) for i in range(cnt)]

        model = cplex.Cplex()
        model.parameters.timelimit.set(600)
        model.parameters.emphasis.memory.set(True)
        model.parameters.mip.tolerances.mipgap.set(1e-5)
            
        # print out
        # model.parameters.mip.display.set(4)
        model.set_log_stream(None)
        model.set_error_stream(None)
        model.set_warning_stream(None)
        model.set_results_stream(None)
            
        start_time = time.time()
            
        model.objective.set_sense(model.objective.sense.maximize)
        model.variables.add(obj=coef_obj, lb=var_lb, ub=var_ub, 
                            types=var_type, names=var_names)
        model.linear_constraints.add(lin_expr=cst, senses=sense, 
                                        rhs=rhs, names=cst_names)
        L = cplex.SparsePair(ind = ["w_other{}".format(i) for i in range(l)] + ["w{}".format(i) for i in range(cnt)] +\
                                    ["w_other{}".format(l+i) for i in range(self.P-l-cnt)], 
                            val = Hw_orig)
        H_flatten = []
        for H_i in H:
            H_flatten += H_i
        w_vars = ["w_other{}".format(i) for i in range(l)] + ["w{}".format(i) for i in range(cnt)] +\
                ["w_other{}".format(l+i) for i in range(self.P-l-cnt)]
        w_vars_repeat_1 = []
        for w_var in w_vars:
            w_vars_repeat_1 += [w_var]*len(w_vars)
        w_vars_repeat_2 = w_vars*len(w_vars)
        Q = cplex.SparseTriple(ind1 = w_vars_repeat_1, 
                            ind2 = w_vars_repeat_2, 
                            val = H_flatten)
        model.quadratic_constraints.add(name = "my_quad", lin_expr = L, quad_expr = Q, rhs = q_rhs, sense = "L")
        
        f_time = time.time()-start_time
        print('seconds formulating problem:', f_time)
            
        model.solve()
        
        s_time = time.time()-f_time - start_time
        print('solving time:', s_time)

        print("Solution status = ", model.solution.get_status(), ":", end=' ')
        print(model.solution.status[model.solution.get_status()])
        print("Solution value  = ", model.solution.get_objective_value())
       
        var = model.solution.get_values()
        abs_w = var[:cnt]
        w = var[cnt:2*cnt]
        w_other = var[3*cnt:]
        w_new = w_other[:l] + w + w_other[l:] 

       
        print("Upper bound of variable importance after optimization:", sum(np.multiply(pr, np.abs(w))))
        # print("w after optimization", w_new)

        w_new = np.array(w_new)
        # print("QC lhs:", (w_new-self.w_orig).T @ self.H @ (w_new - self.w_orig))
        obj = self.check_obj(w_new)
        return w_new, sum(np.multiply(pr, np.abs(w))), s_time
    
    def get_binaries(self, d):
        tmp = np.zeros((2**d, d)).astype('int8')
        step = 2
        for j in range(d):
            tmp[step//2:step, :j] = tmp[0:step//2, :j]
            tmp[step//2:step, j] = 1
            step *= 2
        return tmp
    
    def mcr_plus_lp(self, f, fix=False):
        l, cnt = self.get_f_idx(f)
        pr = np.r_[np.zeros(l), np.array([sum(self.X[:, l+c]/self.N) for c in range(cnt)])]
        pr = np.r_[pr, np.zeros(self.P - l - cnt)]
        pr = pr.reshape(self.P, 1)

        binaries = self.get_binaries(cnt)
        res = np.zeros((2**cnt, self.P))
        res_obj = np.zeros((2**cnt, 1))
        start = time.time()
        for j in range(2**cnt):
            w = cp.Variable(self.P)
            row = -1+2*binaries[j,:]
            tmp = np.zeros((self.P, 1))
            tmp[l:l+cnt, 0] = row
            cst = [w[l+c]*tmp[l+c] >=0 for c in range(cnt)]
            
            if fix:
                prob = cp.Problem(cp.Maximize((pr*tmp).T @ w), 
                        [cp.quad_form(w-self.w_orig, self.H) <= 1] + cst + [w[c] == self.w_orig[c] for c in range(l)] + 
                        [w[c] == self.w_orig[c] for c in range(cnt+l, self.P)]
                        )
            else:
                prob = cp.Problem(cp.Maximize((pr*tmp).T @ w), 
                        [cp.quad_form(w-self.w_orig, self.H) <= 1] + cst #+ [w[c] == self.w_orig[c] for c in range(l)] + 
                        #[w[c] == self.w_orig[c] for c in range(cnt+l, self.P)]
                        )
            prob.solve()

          
            if w.value is not None:
                w_fix = w.value
                # print(0.5*(w_fix - self.w_orig).T @ self.H @ (w_fix - self.w_orig), "ub:", self.ub)
                res[j,:] = w_fix
                res_obj[j,0] = prob.value
            else:
                warnings.warn("solution is None")
        train_time = time.time()-start
        print("time:", train_time)
        # print(res[np.argmax(res_obj), :], np.max(res_obj))
        obj_w = self.check_obj(res[np.argmax(res_obj), :])
        return res[np.argmax(res_obj), :], np.max(res_obj), train_time

    def projection(self, f, w_user):
        # w_user is an array of dimension # steps for feature f
        l, cnt = self.get_f_idx(f)
        w_req = self.w_orig[:l]
        w_req = np.concatenate((w_req, w_user))
        w_req = np.concatenate((w_req, self.w_orig[l+cnt:]))


        I = np.diag(np.ones(self.P))
        w = cp.Variable(self.P)  
        
        prob = cp.Problem(cp.Minimize(cp.quad_form(w-w_req, I)), 
                        [cp.quad_form(w-self.w_orig, self.H) <= 1] + [w[k] == self.w_orig[k] for k in range(l)] + 
                        [w[k] == self.w_orig[k] for k in range(l+cnt, self.P)]
                        )
        prob.solve()
        print("Second solve time:", prob.solver_stats.solve_time)
        # print("\nThe optimal value is", prob.value)
        # print("A solution w is", w.value)
        w_fix = w.value
        # print("qc lhs:", (w_fix-self.w_orig).T @ self.H @ (w_fix-self.w_orig))
        # obj_fix = self.check_obj(w_fix)


        w = cp.Variable(self.P)  
        prob = cp.Problem(cp.Minimize(cp.quad_form(w-w_req, I)), 
                        [cp.quad_form(w-self.w_orig, self.H) <= 1]
                        )
        prob.solve()
        print("Second solve time:", prob.solver_stats.solve_time)
        # print("\nThe optimal value is", prob.value)
        # print("A solution w is", w.value)
        w_all = w.value
        # print("qc lhs:", (w_all-self.w_orig).T @ self.H @ (w_all-self.w_orig))
        obj_all = self.check_obj(w_all)

        return w_req, w_fix, w_all

    def sample_in_ellipsoid(self, H, w_orig, n_samples=10000):
        d = H.shape[0]
        u = np.random.normal(size=(n_samples,d)) # randomly sample iid gaussian
        u = u/(np.linalg.norm(u,axis=1).reshape(-1,1)) # normalize to get uniformly random unit vectors
        r = (np.random.random(size=n_samples))**(1/d) # sample radius (uniformly in a sphere)
        x_ = u * r.reshape(-1,1) # x_ is a uniformly random point in a sphere
        
        lamb, V = np.linalg.eigh(H) # eigen decomposition
        a = np.sqrt(1/lamb) # scaling factor
        dw_samples = ((a*V) @ x_.T).T # transformation to a ellipsoid
        w_samples = dw_samples + w_orig

        return w_samples




