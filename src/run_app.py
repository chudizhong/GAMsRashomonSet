import numpy as np
import pandas as pd
import pickle
import os
from src.rset_app import *
import time
import warnings
import torch
import src.utils as utils
from matplotlib import pyplot as plt



def get_models_from_rset(filepath, n_samples=100, plot_shape=False):
    """
    Input: 
        filepath: string. Store the Rashomon set of a sparse GAM model. 
        n_samples: integer. Sample n_samples models from the Rashomon set.  
        plot_shape: boolean. Default is False. If True, plot the shape function of each variable. 
    """

    rset = RSetGAMs(filepath)
    w_samples = rset.sample_in_ellipsoid(rset.H, rset.w_orig, n_samples=n_samples)

    if plot_shape:
        count = 0
        for key, values in rset.xlabel.items():
            plt.figure(figsize=(5.5,4))
            l = count
            r = count +len(values)-1
            count += len(values)-1

            if key == "bias":
                continue
    
            for i, w_new in enumerate(w_samples):
                plt.step(values, np.hstack((w_new[l], w_new[l:r])), c="red", alpha=0.05)

            plt.step(values, np.hstack((rset.w_orig[l], rset.w_orig[l:r])), c="dimgray", linewidth=2)
            
            plt.xlabel(key, fontsize=22)
            plt.ylabel("predicted logit", fontsize=22)
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)
            # plt.savefig("figures/many_models_{}_{}_{}_{}_{}.png".format(rset.dname, rset.lamb0, rset.lamb2, rset.multiplier, key), bbox_inches='tight')
            plt.show()
    return w_samples



def variable_importance_range(filepath, mip=False, plot_shape=True, plot_vir=True):
    """
    Input: 
        filepath: string. Store the Rashomon set of a sparse GAM model. 
        mip: boolean. Default is False. If True, use mixed integer programming to find the upper bound of variable importance range. 
        plot_shape: boolean. Default is True. If True, plot the shape function of each variable. 
        plot_vir: boolean. Default is True. If True, plot the variable importance range. 
    """
    rset = RSetGAMs(filepath)
    w_c = rset.w_orig

    xlabel = rset.xlabel
    vir_fix = []
    vir_not_fix = []
    count = 0
    for f, values in xlabel.items():
        if f == "bias":
            count += len(values)-1
            continue
        print(f)
        
        print("----------------------variable importance Minus----------------------")
        wl_fix, vil_fix, time_fix, wl, vil, time = rset.mcr_minus(f=f)
        
        if mip:
            
            print("----------------------MCR plus mip----------------------")
            wu_fix, viu_fix, time_fix = rset.mcr_plus_mip(f=f, fix=True)
            wu, viu, time = rset.mcr_plus_mip(f=f, fix=False)
        else:
            print("----------------------variable importance plus LP--------------------")
            wu_fix, viu_fix, time_fix = rset.mcr_plus_lp(f=f, fix=True)
            wu, viu, time = rset.mcr_plus_lp(f=f, fix=False)

        vir_fix.append([vil_fix, viu_fix])
        vir_not_fix.append([vil, viu])

        if plot_shape:
            l = count
            r = count + len(values)-1
            fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,5))
            if values[1] - values[0] <= 0.05*values[-1]:
                values[0] = values[1] - 0.05*values[-1]
        
            y_max = max(w_c[l:r].max(),wl_fix[l:r].max(),wl[l:r].max(),wu_fix[l:r].max(),wu[l:r].max())
            y_min = min(w_c[l:r].min(),wl_fix[l:r].min(),wl[l:r].min(),wu_fix[l:r].min(),wu[l:r].min())
            y_max, y_min = (y_max+y_min)/2 + (y_max-y_min)*0.7, (y_max+y_min)/2 - (y_max-y_min)*0.7

            ax1.step(values, np.hstack((w_c[l], w_c[l:r])), c="dimgray",alpha=0.9, linewidth=3.2, label=r"$\omega_c$")
            ax1.step(values, np.hstack((wl_fix[l], wl_fix[l:r])), c="springgreen",alpha=1, label=r"$\omega_{VI_-}$" + "(fix)")
            ax1.step(values, np.hstack((wl[l], wl[l:r])), c="yellow", alpha=1, label=r"$\omega_{VI_-}$" + "(not fix)")
            ax1.set_ylim([y_min, y_max])

            ax2.step(values, np.hstack((w_c[l], w_c[l:r])), c="dimgray",alpha=0.9, linewidth=3.2, label=r"$\omega_c$")
            ax2.step(values, np.hstack((wu_fix[l], wu_fix[l:r])), c="hotpink", alpha=1, label=r"$\omega_{VI_+}$" + "(fix)")
            ax2.step(values, np.hstack((wu[l], wu[l:r])), c="red", alpha=1, label=r"$\omega_{VI_+}$" + "(not fix)")
            ax2.set_ylim([y_min, y_max])

            ax1.set_title("shape function ("+r"$VI_-$"+")", fontsize=20)
            ax1.set_xlabel(f, fontsize=15)
            ax1.set_ylabel("predicted logit", fontsize=15)
            ax1.legend(fontsize=13)
            ax2.set_title("shape function ("+r"$VI_+$"+")", fontsize=20)
            ax2.set_xlabel(f, fontsize=15)
            ax2.set_ylabel("predicted logit", fontsize=15)
            ax2.legend(fontsize=13)
            ax1.tick_params(axis='both', which='major', labelsize=12)
            ax2.tick_params(axis='both', which='major', labelsize=12)
            plt.tight_layout()
            # plt.savefig("figures/vir_shape_{}_{}_{}_{}_{}.png".format(rset.dname, rset.lamb0, rset.lamb2, rset.multiplier, f), bbox_inches='tight')
            plt.show()
            count += len(values)-1
  
    if plot_vir:
        vir_fix = np.array(vir_fix)
        vir_not_fix = np.array(vir_not_fix)
        P = vir_fix.shape[0]
        plt.figure(figsize=(8, 5))
        plt.hlines(np.arange(P).astype("float") + 0.1, vir_not_fix[:,0], vir_not_fix[:,1], colors="#012169", label="not fix other coefs")
        plt.hlines(np.arange(P).astype("float") - 0.1, vir_fix[:,0], vir_fix[:,1], colors="#4B9CD3", label="fix other coefs")
        plt.yticks(range(P), list(xlabel.keys())[1:], fontsize=15)
        plt.xlabel("variable importance range", fontsize=15)
        plt.tight_layout()
        # plt.savefig("figures/vir_{}_{}_{}_{}.png".format(rset.dname, rset.lamb0, rset.lamb2, rset.multiplier), bbox_inches='tight')
        plt.show()
        



def plot_updated_shape(filepath, f, w_new=None, c=None, label=None, title=None):
    """
    Input: 
        filepath: string. Store the Rashomon set of a sparse GAM model.
        f: string. Feature name.  
        w_new: np array. Specify the new shape function.
        c: string. Specify the color of the shape function. 
        label: string. Specify the label of the shape function. 
        title: string. Specify the title of the plot. 
    """
    rset = RSetGAMs(filepath)
    count = 0
    for key, values in rset.xlabel.items():
        
        l = count
        r = count +len(values)-1
        count += len(values)-1

        if key != f:
            continue
        
        plt.figure()
        plt.step(values, np.hstack((rset.w_orig[l], rset.w_orig[l:r])), c="dimgray", linewidth=2, label=r"$\omega_c$")
        
        if w_new is not None:
            plt.step(values, np.hstack((w_new[l], w_new[l:r])), c=c, label=label)
        
        plt.xlabel(key, fontsize=20)
        plt.ylabel("predicted logit", fontsize=20)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.legend(fontsize=18)
        # plt.savefig("figures/{}_{}_{}_{}_{}_{}.png".format(title, rset.dname, rset.lamb0, rset.lamb2, rset.multiplier, key), bbox_inches='tight')
        plt.show()
        

def get_monotone(filepath, f, direction):
    """
    Input: 
        filepath: string. Store the Rashomon set of a sparse GAM model. 
        f: string. Feature name. 
        direction: string. Specify the monotonicity of the shape function. Use either "increase" or "decrease". 
    """
    print("feature", f)
    rset = RSetGAMs(filepath)
    print("----------------------monotonicity----------------------")
    w = rset.monotonicity(f=f, dir=direction)
    print("updated w", w)
    plot_updated_shape(filepath, f, w_new = w, c = "gold", label=r"$\omega_{new}$", title="monotone")

def get_projection(filepath, f, w_user):
    """
    Input: 
        filepath: string. Store the Rashomon set of a sparse GAM model. 
        f: string. Feature name. 
        w_user: array. Specify the shape function for feature f that the user prefer. 
    """
    print("feature", f)
    rset = RSetGAMs(filepath)
    print("----------------------projection----------------------")
    w_req, w_fix, w_not_fix = rset.projection(f=f, w_user=w_user)
    print("w_fix", w_fix)
    print("w_not_fix", w_not_fix)
    plot_updated_shape(filepath, f, w_new = w_req, c = "red", label=r"$\omega_{req}$", title="proj_req_"+f)
    plot_updated_shape(filepath, f, w_new = w_not_fix, c="lime", label=r"$\omega_{new}$", title="proj_"+f)


def test_jump(filepath, n_samples, i, j, k):
    """
    Input: 
        filepath: string. Store the Rashomon set of a sparse GAM model. 
        n_samples: integer. Sample n_samples models from the Rashomon set.  
        i, j, k: three consecutive integers. Specify the region of jump.  
    """
    rset = RSetGAMs(filepath)
    w_samples = get_models_from_rset(filepath, n_samples=n_samples, plot_shape=False)
    cnt = 0
    for f_idx, w in enumerate(w_samples):
        if (w[i] > w[j] and w[k]> w[j]) or (w[i]<w[j] and w[k] < w[j]):
            cnt += 1       
    print("proportion:", cnt/n_samples)
    return cnt



