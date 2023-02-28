import getopt, yaml
from pathlib                               import Path
from utils.constants                       import ANCIL_FILE
import numpy                               as     np
from math                                  import pi
import numpy.ma                            as     ma
import xarray                              as xr
# import pandas                              as pd
import matplotlib.pyplot                   as plt
import matplotlib.colors as colors
import matplotlib as mpl
import matplotlib.ticker as ticker
from ipykernel.kernelapp                   import IPKernelApp

from utils.utils import read_ancilaries, find_closest_value, find_closest_longitude, get_weights #, get_pressure

degree1 = u'\xb0'

class OnlineDiagnostics():
    def __init__(self, argv):
        try:
            opts, args = getopt.getopt(argv, "hc:a", ["cfg_file=", "add="])
        except getopt.GetoptError:
            print("online_diagnostics.py -c [cfg_file] -a [add]")
            sys.exit(2)
        for opt, arg in opts:
            if opt == "-h":
                print("online_diagnostics.py -c [cfg_file]")
                sys.exit()
            elif opt in ("-c", "--cfg_file"):
                yml_cfgFilenm = arg
            elif opt in ("-a", "--add"):
                pass

        # YAML config file
        self.yml_filename = yml_cfgFilenm
        yml_cfgFile = open(self.yml_filename)
        self.yml_cfg = yaml.load(yml_cfgFile, Loader=yaml.FullLoader)

        self._setup_common(self.yml_cfg)
    
    def _setup_common(self, yml_cfg):
        # Load specifications
        self.simulations  = yml_cfg["simulations"]
        self.data_folder  = yml_cfg["data_folder"]
        self.file_pattern = yml_cfg["file_pattern"]

        ## Model's grid
        self.levels, self.latitudes, self.longitudes = read_ancilaries(Path(ANCIL_FILE))
        self.nlev, self.nlat, self.nlon = len(self.levels), len(self.latitudes), len(self.longitudes)
        self.ngeo                       = self.nlat * self.nlon
        self.lat_weights = np.cos(self.latitudes*pi/180.)

    def readin(
        self,
        simulation,
    ):
        
        data_path = self.data_folder.format(simulation=simulation)
        filenm    = self.file_pattern.format(simulation=simulation)
        data      = xr.open_dataset(Path(data_path, filenm))
        
        return data
        
    def zmDim(
        self,
        simulation,
    ):
        
        data      = self.readin(simulation)
        levs      = data['lev']
        lats      = data['lat']
        time      = data['time']
        
        return levs, lats, time

    def lats_to_sin(
        self,
        lats
    ):
        
        lats_sin   = np.sin(np.array([-90,-30,0,30,90]) * pi/180.)
        lats_ticks = [
            '90'+degree1+'S','30'+degree1+'S','EQ.',
            '30'+degree1+'N','90'+degree1+'N'
        ]
        
        return lats_sin, lats_ticks
    
    
    def var_cos_lat(
        self,
        var,
        lats
    ):

        var = var * np.cos(lats * pi/180.)
        
        return var

    
    
    def zmClimo(
        self,
        simulation,
        variable,
        metric='mean',
        nTime=False,
    ):
        
        data      = self.readin(simulation)
        if not nTime: nTime = data.time.shape[0]
        zmClimo = data[variable].isel(time=[0,nTime-1]).mean(dim='time')
        
        return zmClimo

    def plot_zmClimo(
        self,
        simulation,
        variable,
        ref_simulation=False,
        metric='mean',
        conf=1.96, # 95% conf.int ~1.96*std
        mask=True,
        sin_lats=False,
        nTime=False,
        size=8,
        aspect=1.5,
        interpolation=1,
        ds_spacing=None,
        contour_levels=None,
        zm_center=False,
        extend='both',
        contours=False,
        levels=False,
        save=False,
        savenm=False,
        **kwargs
    ):

        mpl.rcParams['axes.labelsize'] = 28 #'xx-large'
        mpl.rcParams['font.size']      = 24
        
        ds = self.readin(simulation)
        
        levs, lats, time = self.zmDim(simulation)
        if interpolation > 1:
            lats = np.linspace(lats[0], lats[-1], ds.dims["lat"] * interpolation)
            levs = np.linspace(levs[0], levs[-1], ds.dims["lev"] * interpolation)
            ds = ds.interp(lat=lats, lev=levs)
        
        if not nTime: nTime = ds.time.shape[0]
        if metric == 'std':
            zm    = ds[variable].isel(time=slice(0,nTime-1)).std(dim='time', ddof=1)
        else:
            zm    = ds[variable].isel(time=slice(0,nTime-1)).mean(dim='time')
        if zm_center is False: zm_center = zm.mean()
        units = ds[variable].units
        
        if ref_simulation and simulation != 'SPCAM':
            ds_SPCAM = self.readin('SPCAM')
            if interpolation > 1: ds_SPCAM = ds_SPCAM.interp(lat=lats, lev=levs)
            if metric == 'std':
                zm_SPCAM = ds_SPCAM[variable].isel(time=slice(0,nTime-1)).std(dim='time',ddof=1)
            else:
                zm_SPCAM = ds_SPCAM[variable].isel(time=slice(0,nTime-1)).mean(dim='time')
                zm_SPCAM_std = ds_SPCAM[variable].isel(time=slice(0,nTime-1)).std(dim='time',ddof=1)
            # zm_SPCAM = self.zmClimo('SPCAM',variable)
            zm       = zm - zm_SPCAM
            if mask: zm = zm.where(abs(zm) > zm_SPCAM_std*conf)
            zm_center = 0.
        
        if sin_lats:        
            lats = np.sin(lats * pi/180.)
            zm   = zm.assign_coords(lat=("lat", lats.data))
            lats, lats_ticks = self.lats_to_sin(lats)
        
        # # data spacing
        # maximum = np.float(max(abs(zm.min()), abs(zm.max())))
        # vmax=maximum; vmin=-1*maximum
        # ds_spacing = np.linspace(vmin,vmax,11)
        
        cbar_kwargs={
            "label": units,
            "ticks":ds_spacing,
            # "spacing":'proportional',
            "spacing":'uniform',
            "extend":extend,
        }
        zm.plot.pcolormesh(
            # center=zm_center,
            cbar_kwargs=cbar_kwargs,
            size=size,
            aspect=aspect,
            **kwargs
        )
        if contours:
            zm.plot.contour(
                # center=zm_center,
                # levels=contour_levels,
                # colors='k',
                # cbar_kwargs=cbar_kwargs,
                # add_colorbar=True,
                # size=size,
                # aspect=aspect,
                **kwargs
            )

        plt.xlabel('Latitudes')
        plt.ylabel('Pressure (hPa)')
        
        plt.gca().invert_yaxis()
        if sin_lats: plt.xticks(lats, lats_ticks)
        
        plt.tight_layout()
        
        if save:
            Path(save).mkdir(parents=True, exist_ok=True)
            f'SHAP_values_{nn_case}_{clima_case}.png' if not savenm else savenm
            plt.savefig(f"{save}/{savenm}",dpi=1000.,bbox_inches='tight')
            print(f"{save}/{savenm}")

            
    def plot_latitudinals(
        self,
        simulations,
        variables,
        cos_lats=True,
        sin_lats=True,
        std=True,
        conf=1., # 1.96 for 95% conf.int (~1.96*std)
        nTime=False,
        leg=True,
        size=8,
        aspect=1.5,
        save=False,
        savenm=False,
        **kwargs
    ):

        mpl.rcParams['axes.labelsize'] = 'x-large'
        mpl.rcParams['font.size']      = 15
        
        colors_dic = {
            'SPCAM':'k',
            'Non-causalNNCAM':'b',
            'Causally-informedNNCAM':'r',
            'Causally-informed0.59NNCAM':'orange',
        }
        linestyle_dic = {
            'PRECT':'-',
            'FLNT':'-',
            'FSNT':'--',
            'FLNS':'-',
            'FSNS':'--',
        }

        for i, variable in enumerate(variables):
        
            if variable == 'PRECT':
                conv = 86400. * 1.e3 # m/s to mm day-1
            else:
                conv = 1.

            for j, simulation in enumerate(simulations): 

                ds = self.readin(simulation)

                lats, time = ds['PRECT'].coords['lat'], ds['PRECT'].coords['time']
                if not nTime: nTime = ds.time.shape[0]

                var_mean = ds[variable].isel(time=slice(0,nTime-1)).mean(dim='time') * conv
                var_std  = ds[variable].isel(time=slice(0,nTime-1)).std(dim='time', ddof=1) * conv * conf
                var_mean_mstd = var_mean-var_std
                var_mean_pstd = var_mean+var_std
                if variable == 'PRECT':
                    units = 'mm day$^{-1}$'
                else:
                    units = 'W m$^{-2}$'

                if cos_lats: 
                    var_mean = self.var_cos_lat(var_mean, lats)
                    var_std  = self.var_cos_lat(var_std, lats)
                    var_mean_mstd = self.var_cos_lat(var_mean_mstd, lats)
                    var_mean_pstd = self.var_cos_lat(var_mean_pstd, lats)

                if sin_lats:
                    lats_sin = np.sin(lats * pi/180.)
                    var_mean = var_mean.assign_coords(lat=("lat", lats_sin.data))
                    var_std  = var_std.assign_coords(lat=("lat", lats_sin.data))
                    var_mean_mstd = var_mean_mstd.assign_coords(lat=("lat", lats_sin.data))
                    var_mean_pstd = var_mean_pstd.assign_coords(lat=("lat", lats_sin.data))
                    lats, lats_ticks = self.lats_to_sin(lats_sin)

                var_mean.plot(
                    color=colors_dic[simulation],
                    linewidth=[1.5,3][i==0],
                    linestyle=linestyle_dic[variable],
                    label=['',simulation][i==0],
                    **kwargs
                )
                if std and simulation == 'SPCAM':
                    # var_mean_mstd.plot(color='k',linewidth=1,**kwargs)
                    # var_mean_pstd.plot(color='k',linewidth=1,**kwargs)
                    plt.fill_between(lats_sin, var_mean_mstd, var_mean_pstd, color='k', alpha=0.2)
                if std:
                    var_std.plot(color=colors_dic[simulation],linewidth=1.5,linestyle='--',**kwargs)

        plt.xlabel('Latitudes')
        if len(variables) == 1 and variable == 'PRECT':
            ylab = 'Precipitation'
        else:
            ylab = "Net rad. fluxes"
        plt.ylabel(f"{ylab} \n [{units}]")
        
        if sin_lats: plt.xticks(lats, lats_ticks)
        
        if leg: plt.legend(loc=0,frameon=False,fontsize=11)
        
        plt.tight_layout()
        
        if save:
            Path(save).mkdir(parents=True, exist_ok=True)
            f'SHAP_values_{nn_case}_{clima_case}.png' if not savenm else savenm
            plt.savefig(f"{save}/{savenm}",dpi=1000.,bbox_inches='tight')
            print(f"{save}/{savenm}")
            
            