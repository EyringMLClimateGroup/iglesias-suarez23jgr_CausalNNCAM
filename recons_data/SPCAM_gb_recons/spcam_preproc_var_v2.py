#!/usr/bin/env python
# coding: utf-8
"""
HISTORY:
   2021-Feb-09: Distinguish between 2D & 3D variables even for a single lev,      FIS
   2020-Jul-29: paths & running (alloc) updated,                                  FIS
   2020-Jun-26: Written,                                                          FIS

NOTES:
   to run
   (https://www.dkrz.de/up/systems/mistral/running-jobs/slurm-introduction):
      salloc --partition=compute --nodes=1 --time=5:00:00 --account $PROJECT
      ssh <node_allocated>
"""


# **Description**: Re-construct time-(lev)-lat-lon SPCAM data from preprocessed output.
import sys, os, ast, pdb
import xarray        as xr
import numpy         as np
import create_netcdf as nc


# ***Definitions***
def preproc_to_raw(preproc, raw):
    for i in range(len(preproc)):
        
        value    = preproc[i]
        idx_time = time_df[i]
        iLat     = lats_df[i]
        idx_lat  = np.where(lats == iLat)[0][0]
        iLon     = lons_df[i]
        idx_lon  = np.where(lons == iLon)[0][0]
        raw[idx_time, :, idx_lat, idx_lon] = value

    return raw


# Arguments (filemn & variables[list])
filenm       = str(sys.argv[1])
dataPath     = str(sys.argv[2])
outPath      = str(sys.argv[3])
var_all_list = sys.argv[4].split(' ')
count        = sys.argv[5]
##filenm       = '002_train_1_month.nc'
##var_all_list = 'tphystnd'
##print(var_all_list)

# File to be processsed
##filenm     = '003_train_2_month.nc'
##filenm     = '002_train_1_year.nc'
##filenm     = '002_train_1_month.nc'
#filenm     = 'ps_1_year.nc'
#var_all_list = ['ps']

# #### Paths
scriptPath = os.getcwd()
home       = os.popen("echo $HOME").readlines()[0][:-1]
#dataPath   = home+'/work/data/SPCAM_preproc'
#outPath    = home+'/work/data/SPCAM_recons'


# ***SPCAM levels*** (from raw data)
levs = np.array([3.64346569404006, 7.59481964632869, 14.3566322512925,
                 24.6122200042009, 38.2682997733355, 54.5954797416925, 
                 72.0124505460262, 87.8212302923203, 103.317126631737, 
                 121.547240763903, 142.994038760662, 168.225079774857, 
                 197.908086702228, 232.828618958592, 273.910816758871,
                 322.241902351379, 379.100903868675, 445.992574095726, 
                 524.687174707651, 609.778694808483, 691.389430314302, 
                 763.404481112957, 820.858368650079, 859.53476652503, 
                 887.020248919725, 912.644546944648, 936.198398470879,
                 957.485479535535, 976.325407391414, 992.556095123291])


# #### Metadata
units_dict = {'ps':'Pa',
              'solin':'W/m2',
              'shflx':'W/m2',
              'lhflx':'W/m2',
              'fsnt':'W/m2',
              'fsns':'W/m2',
              'flnt':'W/m2',
              'flns':'W/m2',
              'prect':'m/s',
              'prectend':'m/s',
              'precst':'m/s',
              'precsten':'m/s',
              'qbp':'kg/kg',
              'qcbp':'kg/kg',
              'qibp':'kg/kg',
              'tbp':'K',
              'vbp':'m/s',
              'phq':'kg/kg/s',
              'phcldliq':'kg/kg/s',
              'phcldice':'kg/kg/s',
              'tphystnd':'K/s',
              'qrl':'K/s',
              'qrs':'K/s',
              'dtvke':'K/s',
              'qdt_adiabatic':'kg/kg',
              'qcdt_adiabatic':'kg/kg',
              'qidt_adiabatic':'kg/kg',
              'tdt_adiabatic':'K',
              'vdt_adiabatic':'m/s',
             }
ln_dict    = {'ps':'Surface pressure',
              'solin':'Solar insolation',
              'shflx':'Surface sensible heat flux',
              'lhflx':'Surface latent heat flux',
              'fsnt':'Net solar flux at top of model',
              'fsns':'Net solar flux at surface',
              'flnt':'Net longwave flux at top of model',
              'flns':'Net longwave flux at surface',
              'prect':'Total (convective and large-scale) precipitation rate',
              'prectend':'Large-scale (stable) snow rate (water equivalent)',
              'precst':'prectend+precsten',
              'precsten':'Large-scale (stable) snow rate (water equivalent)',
              'qbp':'Specific humidity before physics',
              'qcbp':'Convective portion of QBP',
              'qibp':'Ice phase specific humidity',
              'tbp':'Temperature before physics',
              'vbp':'Meridional wind component',
              'phq':'Specific humidity tendency',
              'phcldliq':'Specific humidity tendency due to cloud liquid particles',
              'phcldice':'Specific humidity tendency due to ice phase cloud particles',
              'tphystnd':'Temperature tendency',
              'qrl':'Long wave heating rate',
              'qrs':'Short wave heating rate',
              'dtvke':'dT/dt vertical diffusion KE dissipation',
              'qdt_adiabatic':'Adiabatic specific humidity tendency after physics',
              'qcdt_adiabatic':'Adiabatic specific humidity tendency after physics due to convection',
              'qidt_adiabatic':'Adiabatic specific humidity tendency after physics due to icy cloud particles',
              'tdt_adiabatic':'Adiabatic temperature tendency after physics',
              'vdt_adiabatic':'Adiabatic meridional wind component tendency after physics',
             }
varnm_3d_list = ['qbp','qcbp','qibp','tbp','vbp','phq','phcldliq','phcldice','tphystnd','qrl','qrs','dtvke',
                 'qdt_adiabatic','qcdt_adiabatic','qidt_adiabatic','tdt_adiabatic','vdt_adiabatic']
varnm_2d_list = ['ps','solin','shflx','fsnt','fsns','flnt','flns','prect','prectend','precst','precsten']


# **Read-in SPCAM pre-processed data**
ds      = xr.open_dataset(dataPath+'/'+filenm)

time_df = ds.time.values[:]
times   = np.array(list(dict.fromkeys(time_df)))
nTimes  = len(times)
#print(time_df.shape)
#print('Time: ', times)

lats_df = ds.lat.values[:]
lats    = np.array(list(dict.fromkeys(lats_df)))
nLats   = len(lats)
#print(lats_df.shape)
#print('Latitudes: ', lats)

lons_df = ds.lon.values[:]
lons    = np.array(list(dict.fromkeys(lons_df)))
nLons   = len(lons)
#print(lons_df.shape)
#print('Longitudes: ', lons)

#print('variables: ', ds.variables)
print('vars: ', ds.vars)

# #### Sort data by variables (3-D & 2-D)
var_names = ds.var_names.values[:]
var_list = list(dict.fromkeys(var_names))
print('Number of variables: ', len(var_list))
print('List of variables: ', var_list)

var_idx_dict = {i.lower() : np.where(var_names == i)[0] for i in var_list}
#print('Variables indexes: ', var_idx_dict)

ds_vars   = ds.vars.values[:]
#print('Dataset shape: ', ds_vars.shape)

ds_vars_dict = {i.lower():ds_vars[:,var_idx_dict[i.lower()][0]:var_idx_dict[i.lower()][-1]+1]         for i in var_list}
#print('QBP shape (3-D): ', ds_vars_dict['qbp'].shape)
#print('PRECT shape (2-D): ', ds_vars_dict['prect'].shape)


# #### 3-D variables (zero arrays)
#var_3d_dict = {i.lower() : np.zeros([nTimes+1,len(var_idx_dict[i.lower()]),nLats,nLons])                for i in var_list if len(var_idx_dict[i.lower()]) > 1}
var_3d_dict = {i.lower() : np.zeros([nTimes+1,len(var_idx_dict[i.lower()]),nLats,nLons])  for i in var_list if len(var_idx_dict[i.lower()]) > 1 or i.lower() in varnm_3d_list}
var_3d_dict.keys()

# #### 2-D variables (zero arrays)
var_2d_dict = {i.lower() : np.zeros([nTimes+1,len(var_idx_dict[i.lower()]),nLats,nLons])  for i in var_list if len(var_idx_dict[i.lower()]) == 1 and i.lower() in varnm_2d_list}
var_2d_dict.keys()


# #### Fill-in zero arrays & Create netCDF files
var_2d_list  = var_2d_dict.keys()
var_3d_list  = var_3d_dict.keys()
#var_all_list = list(var_2d_dict.keys()) + list(var_3d_dict.keys())
for iKey in var_all_list:

    var_list    = [var_3d_list,var_2d_list][iKey in var_2d_list]
    var_dict    = [var_3d_dict,var_2d_dict][iKey in var_2d_list]
    
    outFile     = iKey+'_'+filenm
    exists_file = os.path.isfile('%s/%s' % (outPath,outFile))

    if not exists_file:
        print('Processing ', iKey)
        preproc           = ds_vars_dict[iKey]
        raw               = var_dict[iKey]
        var_dict[iKey]    = preproc_to_raw(preproc, raw)
        print(var_dict[iKey].shape)

        if iKey in var_2d_list:
            nc.create_netcdf(var1=var_dict[iKey][:,0,:,:],var1N=iKey,
                             var1U=units_dict[iKey],var1LN=ln_dict[iKey],
                             times=times,
                             lats=lats,
                             lons=lons,
                             fname=outFile,
                             outPath=outPath)
        elif iKey in var_3d_list and len(var_idx_dict[iKey]) == 1:
            nc.create_netcdf(var1=var_dict[iKey],var1N=iKey,
                             var1U=units_dict[iKey],var1LN=ln_dict[iKey],
                             times=times,
                             levs=[levs[int(count)-1]],
                             levU='hPa',
                             lats=lats,
                             lons=lons,
                             fname=outFile,
                             outPath=outPath)
        else:
            nc.create_netcdf(var1=var_dict[iKey],var1N=iKey,
                             var1U=units_dict[iKey],var1LN=ln_dict[iKey],
                             times=times,
                             levs=levs,levU='hPa',
                             lats=lats,
                             lons=lons,
                             fname=outFile,
                             outPath=outPath)
            
    else:
        print(iKey, ' exists, skipping...')
        
    print(); print()
