import numpy as np
import pandas as pd
import pickle
from src.rset_opt import *

# if the sparse GAM model is available in the pickle file, 
# we can directly use it and find it's Rashomon set. 

# filepath stores the sparse GAM that we want to find the Rashomon set for. 
filepath = "diabetes_0.001_0.001_1.01.p"

model = RSetOPT(filepath)
model.finetune_ellipsoid()
model.get_precision()
H_opt = model.get_normalized_H()
w_opt = model.w_orig

# store optimized hessian and w to the file
model.update_file(H_opt, w_opt)



# ----------------------------------------------------
# ----------------------------------------------------
# if the sparse GAM model is not available, 
# we can first get a sparse GAM model and then find it's Rashomon set. 

"""
from src.prepare_gam import *
dname = "diabetes"
lamb0 = 0.001
lamb2 = 0.001
multiplier = 1.01
filepath = prepare_sparse_gam(dname, lamb0, lamb2, multiplier)
"""


# once the pickle file is ready, run line 12-19 to find the Rashomon set. 