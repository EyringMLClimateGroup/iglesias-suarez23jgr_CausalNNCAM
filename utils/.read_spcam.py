#!/usr/local/bin/python
"""
HISTORY:
14-May-2020 Written,                                      FIS

DESCRIPTION:
Read-in SPCAM re-constructed data.

NOTES:

"""




######################### PACKAGES #########################

import sys, os; sys.dont_write_bytecode = True
import glob
from   netCDF4                            import Dataset

############################################################





######################### OPTIONS ##########################

# Paths
home       = os.popen("echo $HOME").readlines()[0][:-1]
dataPath   = home+'/SPCAM_recons'

## SCRIPT
#
# Version
vScript = '(written): 1.0'
# Name
nScript = 'read_spcam.py'

############################################################





######################## DEFINITIONS #######################

def read_spcam(variable, filestr, *, path=dataPath):

    '''
    Takes a variable and filestr (i.e. experiment). 
    Optional specific path.
    Returns the variable.
    '''

    filenm      = path+'/%s_%s.nc' %(variable,filestr)
    ncfile = Dataset(filenm,'r')
    outVar = ncfile.variables[variable][:]
    ncfile.close()
    return outVar

############################################################















