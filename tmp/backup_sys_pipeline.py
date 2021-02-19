#!/usr/bin/env python
# coding: utf-8

# # ***DESCRIPTION*** 
# ## ***Run Tigramite (PCMCI) for SPCAM data with specified settings:***
# ### Fixed:
# - PC-stable (i.e., MCI component not run)
# - tau_min/tau_max = -1
# - Significance: analytics
# - experiments: '002_train_1_year'
# - links: parents (state fields) -> children (parameterizations)
# ### Options:
# - region: lat/lon limits (gridpoints to be used)
# - levels: children's levels to be explored
# - pc_alpha: list of value(s)

# In[14]:


# Python packages
import sys, getopt
import numpy                  as np
from pathlib              import Path

# ## Variables to be processed

# In[ ]:

argv           = sys.argv[1:]
print (argv)
spcam_parents  = ['tbp','qbp','vbp','ps','solin','shflx','lhflx']
spcam_children = ['tphystnd','prect', 'fsns', 'flns']
pc_alphas      = [0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2]
region         = None
lim_levels     = None
target_levels  = None
try:
    opts, args = getopt.getopt(argv,"hp:c:a:r:l:t",["parents=","children=",
                                                    "pc_alphas=","region=",
                                                    "lim_levels=","target_levels="])
except getopt.GetoptError:
    print ('pipeline.py -p [parents] -c [children] -a [pc_alphas] -r [region] -l [lim_levels] -t [target_levels]')
    sys.exit(2)
for opt, arg in opts:
    print(opt, arg)
    if opt == '-h':
        print ('pipeline.py -p [parents] -c [children]')
        sys.exit()
    elif opt in ("-p", "--parents"):
        spcam_parents = arg
    elif opt in ("-c", "--children"):
        spcam_children = arg
    elif opt in ("-a", "--pc_alphas"):
        pc_alphas = arg
    elif opt in ("-r", "--region"):
        region = arg
    elif opt in ("-l", "--lim_levels"):
        lim_levels = arg
    elif opt in ("-t", "--target_levels"):
        target_levels = arg
print ('Parents are: ', spcam_parents)
print ('Children are: ', spcam_children)
print ('PC-alphas are: ', pc_alphas)
print ('Region is: ', region)
print ('lim_levels are: ', lim_levels)
print ('target_levels are: ', target_levels)


