#!/usr/bin/env python

# Read in Yao et al. IRRMIP hydrological cycle components.

# Data from:

#https://figshare.com/articles/dataset/Yao_et_al_Irrigation-induced_land_water_depletion_aggravated_by_climate_change/29664485/1?file=56627927
#SOME FILES MISSING.

# Try

# https://figshare.com/articles/dataset/Yao_et_al_2025_Irrigation-induced_land_water_depletion_aggravated_by_climate_change/28624352/1?file=53106884

# SOME DATA ALSO MISSING HERE, SO HAVE TO USE BOTH.

# Yao code segments used (See below) from :

# https://github.com/YiYao1995/Yao_et_al_2025_Irrigation-induced_land_water_depletion_aggravated_by_climate_change.git



# Mercury local version. CHANGE DIRECTORY NAMES TO WHERE YAO DATA IS DOWNLOADED TO.

# To run: exec(open('irrmip_proc.py').read())


import netCDF4 as nc
import numpy as np
#from Load_data import Data_from_nc
import scipy.io as scio
import matplotlib.pyplot as plt
from regolsfunc import regols

### 1) CODE BLOCKS TAKEN FROM water_bars_multi_models.ipynb in original Yao et al. 2025 codebase.

# ─── I/O Helpers
────────────────────────────────────────────────────────────

def get_data_from_mat_for_calcu(file, variable):
     var_dict = scio.loadmat(file)
     var = var_dict[variable]
     var = var[:, 29:]
     return var.T

def get_data_from_nc(file, variable):
     file_obj = nc.Dataset(file)
     data = file_obj.variables[variable]
     var_data = np.array(data)
     var_data = var_data[:, 29:, :]
     var_data[var_data > 1000000] = np.nan
     var_data = np.squeeze(var_data)
     return var_data

def get_data_from_mat(file, variable):
     var_dict = scio.loadmat(file)
     var = var_dict[variable]
     var = var[:, 29:]
     return var

## 1a) CHANGE DIRECTORY HERE. For str_base. Also change str_use in every case to directory for original Yao data.

#str_base = '/home/fhl202/data/yao_data' # base directory where code is for averaging / plotting and data. ORIGINAL ZIP FILE.
str_base = '/home/fhl202/data/yao_7z_data' # And where data is. (File link sent on 17/10/2025

#ar6_region = get_data_from_mat_for_calcu(str_base +'/plotting_tools/ar6_region.mat','ar6_region') # ORIGINAL
#AREA = get_data_from_mat(str_base + '/plotting_tools/AREA.mat', 'AREA')
# Area for CLM_data
#irr_diff = get_data_from_mat_for_calcu(str_base +'/plotting_tools/irr_diff_out.mat', 'irr_diff_out')

ar6_region = get_data_from_mat_for_calcu(str_base +'/ar6_region.mat','ar6_region') # 17/10, (Different directory structure.)
AREA = get_data_from_mat(str_base + '/AREA.mat', 'AREA') # Area for CLM_data
irr_diff = get_data_from_mat_for_calcu(str_base + '/irr_diff_out.mat','irr_diff_out')




# ─── Model‐specific loader
──────────────────────────────────────────────────
# Here we have a function for each ESM to read the variable
# This is because the folders and the names are all model-specific
def get_data_cesm2(variable):

#    str_start = '/water_fluxes/CESM2/CESM2_' # original
#    str_start = '/Water_variables/CESM2/CESM2_' # 17/10. *** DOESN'T WORK. HACK BACK IN ORIGINAL.

#    str_use = str_base + str_start

     str_use = '/home/fhl202/data/yao_data/water_fluxes/CESM2/CESM2_' # Original hacked in. 


     str_mid = '_1901_2014_'
     str_end = '_yearmean'

     data_irr01 = get_data_from_nc(str_use + 'IRR01' + str_mid + variable + str_end, variable)
     data_noi01 = get_data_from_nc(str_use + 'NOI01' + str_mid + variable + str_end, variable)

     data_irr02 = get_data_from_nc(str_use + 'IRR02' + str_mid + variable + str_end, variable)
     data_noi02 = get_data_from_nc(str_use + 'NOI02' + str_mid + variable + str_end, variable)

     data_irr03 = get_data_from_nc(str_use + 'IRR03' + str_mid + variable + str_end, variable)
     data_noi03 = get_data_from_nc(str_use + 'NOI03' + str_mid + variable + str_end, variable)

     data_irr = (data_irr01 + data_irr02 + data_irr03) / 3 * 365 * 86400
     data_noi = (data_noi01 + data_noi02 + data_noi03) / 3 * 365 * 86400

     return data_irr[:-1,:,:], data_noi[:-1,:,:]

def get_data_e3sm(variable3):

#    str_start = '/water_fluxes/E3SMv2/E3SM_' # original
     str_start = '/Water_variables/E3SMv2/E3SM_' # 17/10
     str_mid = '_1901_2014_'
     str_end = '_yearmean_0.9x1.25'

     data_irr01 = get_data_from_nc(str_base + str_start + 'IRR01' + str_mid + variable3 + str_end, variable3)
     data_noi01 = get_data_from_nc(str_base + str_start + 'NOI01' + str_mid + variable3 + str_end, variable3)

     data_irr02 = get_data_from_nc(str_base + str_start + 'IRR02' + str_mid + variable3 + str_end, variable3)
     data_noi02 = get_data_from_nc(str_base + str_start + 'NOI02' + str_mid + variable3 + str_end, variable3)

     data_irr = (data_irr01 + data_irr02) / 2 * 365 * 86400
     data_noi = (data_noi01 + data_noi02) / 2 * 365 * 86400

     return data_irr[6:-1,:,:], data_noi[6:-1,:,:]


def get_data_cesm2_gw(variable):

#    str_start = '/water_fluxes/CESM2_gw/CESM2_gw_' # original
#    str_start = '/Water_variables/CESM2_gw/CESM2_gw_' # 17/10. ****DOESN'T WORK. HACK IN ORIGINAL. ****

#    str_use = str_base + str_start

     str_use = '/home/fhl202/data/yao_data/water_fluxes/CESM2_gw/CESM2_gw_' # Originalhacked in.

     str_mid = '_1901_2014_'
     str_end = '_yearmean'

     data_irr01 = get_data_from_nc(str_use + 'IRR01' + str_mid + variable + str_end, variable)
     data_noi01 = get_data_from_nc(str_use + 'NOI01' + str_mid + variable + str_end, variable)

     data_irr02 = get_data_from_nc(str_use + 'IRR02' + str_mid + variable + str_end, variable)
     data_noi02 = get_data_from_nc(str_use + 'NOI02' + str_mid + variable + str_end, variable)

     data_irr03 = get_data_from_nc(str_use + 'IRR03' + str_mid + variable + str_end, variable)
     data_noi03 = get_data_from_nc(str_use + 'NOI03' + str_mid + variable + str_end, variable)

     data_irr = (data_irr01 + data_irr02 + data_irr03) / 3 * 365 * 86400
     data_noi = (data_noi01 + data_noi02 + data_noi03) / 3 * 365 * 86400

     return data_irr[:-1,:,:], data_noi[:-1,:,:]


def get_data_noresm(variable):

#    str_start = '/water_fluxes/NorESM2/NorESM_' # original
#    str_start = '/Water_variables/NorESM/NorESM_' # 17/10 . EVAPORATION MISSING

     # str_use = str_base + str_start

     str_use = '/home/fhl202/data/yao_data/water_fluxes/NorESM2/NorESM_'

     str_mid = '_1901_2014_'
     str_end = '_yearmean'

     data_irr01 = get_data_from_nc(str_use + 'IRR01' + str_mid + variable + str_end, variable)
     data_noi01 = get_data_from_nc(str_use + 'NOI01' + str_mid + variable + str_end, variable)

     data_irr02 = get_data_from_nc(str_use + 'IRR02' + str_mid + variable + str_end, variable)
     data_noi02 = get_data_from_nc(str_use + 'NOI02' + str_mid + variable + str_end, variable)

     data_irr03 = get_data_from_nc(str_use + 'IRR03' + str_mid + variable + str_end, variable)
     data_noi03 = get_data_from_nc(str_use + 'NOI03' + str_mid + variable + str_end, variable)

     data_irr01_temp = np.zeros([116,163,288])# the first year of ensemble 01 is missing
     data_irr01_temp[1:, :, :] = data_irr01
     data_irr01_temp[0, :, :] = (data_irr02[0, :, :] + data_irr03[0, :,:])/2
     data_irr01 = data_irr01_temp

     data_irr = (data_irr01 + data_irr02 + data_irr03) / 3 * 365 * 86400
     data_noi = (data_noi01 + data_noi02 + data_noi03) / 3 * 365 * 86400

     return data_irr[1:-1,:,:], data_noi[1:-1,:,:]


def get_data_ipsl(variable2):

#    str_start = '/water_fluxes/IPSL-CM6/' # original
     str_start = '/Water_variables/IPSL-CM6/' # 17/10

     str_end = '_1901_2014_Month.nc_yearmean_0.9x1.25'

     data_irr = get_data_from_nc(str_base + str_start + 'IRR01_'+ variable2 + str_end, variable2)
     data_noi = get_data_from_nc(str_base + str_start + 'NOI01_'+ variable2 + str_end, variable2)


     data_irr = data_irr * 365 * 86400
     data_noi = data_noi * 365 * 86400

     return data_irr, data_noi

def get_data_cnrm(variable2):

#    str_start = '/water_fluxes/CNRM-CM6-1/' # original
#    str_start = '/Water_variables/CNRM-CM6-1/' # 17/10. *** Most data missing here !!!! ***

#    str_use = str_base + str_start # Add to base.

     str_use = '/home/fhl202/data/yao_data/water_fluxes/CNRM-CM6-1/' # Hack back in original data here...

     str_end = '.nc_yearmean_yearmean_0.9x1.25'


     data_irr01 = get_data_from_nc(str_use + variable2 + '_IRR01' + str_end, variable2)
     data_noi01 = get_data_from_nc(str_use + variable2 + '_NOI01' + str_end, variable2)

     data_irr02 = get_data_from_nc(str_use + variable2 + '_IRR02' + str_end, variable2)
     data_noi02 = get_data_from_nc(str_use + variable2 + '_NOI02' + str_end, variable2)

     data_irr03 = get_data_from_nc(str_use + variable2 + '_IRR03' + str_end, variable2)
     data_noi03 = get_data_from_nc(str_use + variable2 + '_NOI03' + str_end, variable2)

     data_irr04 = get_data_from_nc(str_use + variable2 + '_IRR04' + str_end, variable2)
     data_noi04 = get_data_from_nc(str_use + variable2 + '_NOI04' + str_end, variable2)

     data_irr05 = get_data_from_nc(str_use + variable2 + '_IRR05' + str_end, variable2)
     data_noi05 = get_data_from_nc(str_use + variable2 + '_NOI05' + str_end, variable2)


     data_irr = (data_irr01+data_irr02+data_irr03+data_irr04+data_irr05) / 5 * 365 * 86400
     data_noi = (data_noi01+data_noi02+data_noi03+data_noi04+data_noi05) / 5 * 365 * 86400

     return data_irr, data_noi


def get_data_miroc(variable):

#    str_start = '/water_fluxes/MIROC-INTEG-ES/' # original
     str_start = '/Water_variables/MIROC-INTEG-ES/' # 17/10

     str_mid = '_mon_MIROC_'
     str_end = '_1901-2014.nc_0.9x1.25_yearmean'

     data_irr01 = get_data_from_nc(str_base + str_start + variable + str_mid + 'IRR01' + str_end, variable)
     data_noi01 = get_data_from_nc(str_base + str_start + variable + str_mid + 'NOI01' + str_end, variable)

     data_irr02 = get_data_from_nc(str_base + str_start + variable + str_mid + 'IRR02' + str_end, variable)
     data_noi02 = get_data_from_nc(str_base + str_start + variable + str_mid + 'NOI02' + str_end, variable)

     data_irr03 = get_data_from_nc(str_base + str_start + variable + str_mid + 'IRR03' + str_end, variable)
     data_noi03 = get_data_from_nc(str_base + str_start + variable + str_mid + 'NOI03' + str_end, variable)

     data_irr = (data_irr01 + data_irr02 + data_irr03) / 3 * 365 * 86400
     data_noi = (data_noi01 + data_noi02 + data_noi03) / 3 * 365 * 86400

     return data_irr, data_noi



# Now into code that actually does stuff? (Rather than functions.)

print('Loading data...')

MODEL_LOADERS = {
     'CESM2':    get_data_cesm2,
     'CESM2_gw': get_data_cesm2_gw,
     'NorESM':   get_data_noresm,
     'E3SM':     get_data_e3sm, # DATA ABSENT
     'IPSL':     get_data_ipsl, # DATA ABSENT
     'CNRM':     get_data_cnrm,
     'MIROC':    get_data_miroc,
}


# specify per-model which variable names to fetch
# tuple of (rain_var, snow_var) means load both and sum
# single-member tuple means loader returns P directly
PREC_VARS = {
     'CESM2':    ('RAIN_FROM_ATM',  'SNOW_FROM_ATM'),
     'CESM2_gw': ('RAIN_FROM_ATM',  'SNOW_FROM_ATM'),
     'NorESM':   ('RAIN_FROM_ATM',  'SNOW_FROM_ATM'),
     'E3SM':     ('RAIN',           'SNOW'),
     'IPSL':     ('pr',),    # loader returns P already
     'CNRM':     ('pr',),
     'MIROC':    ('pr',),
}

# container for all precipitation datasets
precip = {}

for model, vars_ in PREC_VARS.items():
     loader = MODEL_LOADERS[model]
     # if we have separate rain & snow variables, fetch both & sum
     if len(vars_) == 2:
         rain_var, snow_var = vars_
         irr_r = loader(rain_var)[0]  # [0] = IRR
         noi_r = loader(rain_var)[1]  # [1] = NOI
         irr_s = loader(snow_var)[0]
         noi_s = loader(snow_var)[1]

         precip[model] = {
             'IRR': irr_r + irr_s,
             'NOI': noi_r + noi_s
         }
     else:
         # single var: loader returns precipitation directly
         (var,) = vars_
         irr_p, noi_p = loader(var)
         precip[model] = {
             'IRR': irr_p,
             'NOI': noi_p
         }

# Example access:
# IWW_IRR_1901_2014_CESM2_P   = precip['CESM2']['IRR']
# IWW_NOI_1901_2014_CESM2_P   = precip['CESM2']['NOI']
# IWW_IRR_1901_2014_E3SM_P    = precip['E3SM']['IRR']
# IWW_NOI_1901_2014_MIROC_P   = precip['MIROC']['NOI']

RUNOFF_VARS = {
     'CESM2':    'QRUNOFF',
     'CESM2_gw': 'QRUNOFF',
     'NorESM':   'QRUNOFF',
     'E3SM':     'QRUNOFF',
     'IPSL':     'mrro',
     'CNRM':     'mrro',
     'MIROC':    'mrro',
}

# container for all runoff datasets
runoff = {}

for model, var in RUNOFF_VARS.items():
     loader = MODEL_LOADERS[model]
     irr_r, noi_r = loader(var)
     runoff[model] = {
         'IRR': irr_r,
         'NOI': noi_r
     }


ET_COMPONENTS = {
     'CESM2':    ['QFLX_EVAP_TOT'],
     'CESM2_gw': ['QFLX_EVAP_TOT'],
     'NorESM':   ['QFLX_EVAP_TOT'],
     'E3SM':     ['QSOIL', 'QVEGE', 'QVEGT'],
     'IPSL':     ['evspsbl'],
     'CNRM':     ['evspsbl'],
     'MIROC':    ['evspsbl', 'tran'],
}

evapotran = {}

for model, comps in ET_COMPONENTS.items():
     loader = MODEL_LOADERS[model]
     irr_parts = []
     noi_parts = []

     # load each component and accumulate
     for var in comps:
         irr_var, noi_var = loader(var)
         irr_parts.append(irr_var)
         noi_parts.append(noi_var)

     # sum all parts to get total ET
     total_irr = sum(irr_parts)
     total_noi = sum(noi_parts)

     evapotran[model] = {
         'IRR': total_irr,
         'NOI': total_noi
     }

# specify the TWS variable for each model
TWS_VARS = {
     'CESM2':    'TWS',
     'CESM2_gw': 'TWS',
     'NorESM':   'TWS',
     'E3SM':     'TWS',
     'IPSL':     'mrtws',
     'CNRM':     'mrtws',
     'MIROC':    'tws',
}

# container for all TWS datasets
tws = {}

for model, var in TWS_VARS.items():
     loader = MODEL_LOADERS[model]
     irr_tws, noi_tws = loader(var)
     tws[model] = {
         'IRR': irr_tws,
         'NOI': noi_tws
     }


### 2. NOW MAKE FIGURE 1 IN LAMBERT N_AND_V.

# Data loaded at this point. Now process.

print('Processing...')

DATA_GROUPS = {
     'P'  : precip,
     'R'  : runoff,
     'TWS': tws,
     'ET' : evapotran,
}

# Here we use CESM2 output for masking in case some models also output grid cells over the ocean

for name, group in DATA_GROUPS.items():
     # make mask from CESM2 IRR
     ref_mask = np.isnan(group['CESM2']['IRR'])
     # apply to every other model & both scenarios
     for model, scen_dict in group.items():
         if model == 'CESM2':
             continue
         for scen in ('IRR','NOI'):
             scen_dict[scen][ref_mask] = np.nan



# Now back to defining functions again. ALSO FROM WATER_BARS_MULTI_MODELS.IPYNB

def calcu_global_water(data): # This is the function to calculate global
mean water fluxes (not used here in this script)

#    str_area ='/dodrio/scratch/projects/2022_200/project_output/cesm/yi_yao_IRRMIP/input_data/AREA.mat'
     str_area = '/home/fhl202/data/yao_data/plotting_tools/AREA.mat'

     area = get_data_from_mat(str_area, 'AREA')

     area_for_calcu = area.T

     area_for_calcu[np.isnan(data[0, :, :])] = np.nan


     data_globe = data * AREA.T

     data_globe = np.nansum(data_globe, axis=(1, 2))

     area_for_calcu = np.nansum(area_for_calcu, axis=(0, 1))

     return data_globe/area_for_calcu


def calcu_water_region(data, region_id):

     # This is the function to calculate regional mean water fluxes
(used here)

#    str_area = '/dodrio/scratch/projects/2022_200/project_output/cesm/yi_yao_IRRMIP/input_data/AREA.mat'
     str_area = '/home/fhl202/data/yao_data/plotting_tools/AREA.mat'

     area = get_data_from_mat(str_area, 'AREA')

     # Area for CLM_data, we read it every time when calling this function
     # Because in Jupyter notebook the array will be changed even though only being used in another function

     area_for_calcu = area.T

     # build a mask of all points to set to NaN
     mask = (
         np.isnan(data[0, :, :]) # missing data
         | (np.abs(ar6_region - region_id) > 0.2)    # outside
         [region_id–tol, region_id+tol]
     )

     area_for_calcu[mask] = np.nan

     # Keep only the area in the region

     data_region = data * area_for_calcu

     data_region[:, mask] = np.nan

     data_region = np.nansum(data_region, axis=(1, 2))

     area_for_calcu = np.nansum(area_for_calcu, axis=(0, 1))

     return data_region/area_for_calcu



def get_regional_p_et(region_id):

     def _calc_p_et_anomalies(model_key, region_id, baseline_end=114):
         """
         Returns (irr_anomaly, noi_anomaly) time‐series for the given model.
         """
         # compute raw P–ET and aggregate over the region

#        irr = calcu_water_region(
#            precip[model_key]['IRR'] - evapotran[model_key]['IRR'],
#            region_id
#        )

#        noi = calcu_water_region(
#            precip[model_key]['NOI'] - evapotran[model_key]['NOI'],
#            region_id
#        )

         # compute raw P and ET separately as an alternative

         irr_p = calcu_water_region(precip[model_key]['IRR'],region_id)
         irr_e = calcu_water_region(evapotran[model_key]['IRR'],region_id)

         noi_p = calcu_water_region(precip[model_key]['NOI'],region_id)
         noi_e = calcu_water_region(evapotran[model_key]['NOI'],region_id)

         return irr_p, irr_e, noi_p, noi_e # (No baseline subtracted).

         # baseline from the first `baseline_end` timesteps of the non‑irrigated run
#       baseline = np.mean(noi[:baseline_end])


#       return irr - baseline, noi - baseline  # NOTE WE WILL REMOVE BASELINE OUTSIDE THE ROUTINE AS REQUIRED.



     # 1. Define the models in the exact order you want them stacked
     models = [
         'CESM2',
         'CESM2_gw',
         'NorESM',
         'E3SM',
         'IPSL',
         'MIROC',
         'CNRM',
     ]

     # 2. Prepare empty lists to collect each model's IRR/NOI series
     irr_p_list = []
     irr_e_list = []
     noi_p_list = []
     noi_e_list = []

     # 3. Loop once, calculate both series (using your helper), and append
     for m in models:
         irr_p_ts, irr_e_ts, noi_p_ts, noi_e_ts = _calc_p_et_anomalies(m, region_id)
         irr_p_list.append(irr_p_ts)
         irr_e_list.append(irr_e_ts)
         noi_p_list.append(noi_p_ts)
         noi_e_list.append(noi_e_ts)

     # 4. Stack them all in one go – same as your vstack calls
     data_irr_p = np.vstack(irr_p_list)
     data_noi_p = np.vstack(noi_p_list)
     data_irr_e = np.vstack(irr_e_list)
     data_noi_e = np.vstack(noi_e_list)

     # 5. Compute the difference
     data_diff_p = data_irr_p - data_noi_p
     data_diff_e = data_irr_e - data_noi_e

     return data_irr_p, data_irr_e, data_noi_p, data_noi_e, data_diff_p,data_diff_e


### 3) MAKE FIGURE

# Now back to main program. Diverge from Yao code here and just try to
# retrieve the annual mean time series themselves.

# 1) map your region names → AR6 IDs

REGIONS = {
     'SAS': 38,
     'MED': 20,
     'CNA':  5,
     'WCA': 33,
}

irr_p_arr = np.zeros((4,7,114)) # Region, model, years
noi_p_arr = np.zeros((4,7,114)) # Precipitation
diff_p_arr = np.zeros((4,7,114))
irr_e_arr = np.zeros((4,7,114)) # Region, model, years
noi_e_arr = np.zeros((4,7,114)) # Evaporation
diff_e_arr = np.zeros((4,7,114))


for i, rid in enumerate(REGIONS.items()):
     irr_p, irr_e, noi_p, noi_e, diff_p, diff_e = get_regional_p_et(rid[1])
     irr_p_arr[i,:,:] = irr_p # Allocate data to arrays. P first.
     noi_p_arr[i,:,:] = noi_p
     diff_p_arr[i,:,:] = diff_p
     irr_e_arr[i,:,:] = irr_e # Allocate data to arrays. E first.
     noi_e_arr[i,:,:] = noi_e
     diff_e_arr[i,:,:] = diff_e


# Make six 19 year means for each variable as Yao did.

noi_p_arr_19 = np.reshape(noi_p_arr,(4,7,6,19)).mean(3)
noi_e_arr_19 = np.reshape(noi_e_arr,(4,7,6,19)).mean(3)
irr_p_arr_19 = np.reshape(irr_p_arr,(4,7,6,19)).mean(3)
irr_e_arr_19 = np.reshape(irr_e_arr,(4,7,6,19)).mean(3)


# Sketch plots

nmodels = np.shape(irr_p_arr)[1] # number of models

psyms = ['.','x','>','+','<','2','*'] # plotting symbols
#noicols = ['cyan','turquoise','blue','blue','blue']  # Colors for No
irrigation change
#irrcols = ['hotpink','lightcoral','red','red','red']
modnames =
['CESM2','CESM2_gw','NorESM2','E3SM','IPSL','MIROC-INTEG-ES','CNRM-CM6-1']
regnames = ['SAS','MED','CNA','WCA']
regnamesfull = ['South Asia','Mediterranean','Central North
America','West Central Asia']


plt.clf()
plt.figure()

# Top panel. South Asia

regnum = 0 # which region?

plt.subplot(1,2,1)

for i in range(nmodels):
plt.scatter(noi_p_arr_19[regnum,i,0],noi_e_arr_19[regnum,i,0],marker='s',edgecolors='deepskyblue',color='none',s=80)
# box round outside for first point
plt.scatter(noi_p_arr_19[regnum,i,-1],noi_e_arr_19[regnum,i,-1],marker='o',edgecolors='deepskyblue',color='none',s=80)
plt.scatter(noi_p_arr_19[regnum,i,:],noi_e_arr_19[regnum,i,:],marker=psyms[i],color='deepskyblue')
# points
plt.scatter(irr_p_arr_19[regnum,i,0],irr_e_arr_19[regnum,i,0],marker='s',edgecolors='maroon',color='none',s=80)
plt.scatter(irr_p_arr_19[regnum,i,-1],irr_e_arr_19[regnum,i,-1],marker='o',edgecolors='maroon',color='none',s=80)
plt.scatter(irr_p_arr_19[regnum,i,:],irr_e_arr_19[regnum,i,:],marker=psyms[i],color='maroon',label=modnames[i])


plt.xlabel(r'$P$ [mm yr$^{-1}$]')
plt.ylabel(r'$ET$ [mm yr$^{-1}$]')
plt.title(regnamesfull[regnum])
plt.legend(fontsize='small')


# Bottom panel. Mediterranean

regnum=1

plt.subplot(1,2,2)

plt.plot([200.,800.],[200.,800.],'k:') # Have ET = P line.

for i in range(nmodels):
plt.scatter(noi_p_arr_19[regnum,i,0],noi_e_arr_19[regnum,i,0],marker='s',edgecolors='deepskyblue',color='none',s=80)
# box round outside for first point
plt.scatter(noi_p_arr_19[regnum,i,-1],noi_e_arr_19[regnum,i,-1],marker='o',edgecolors='deepskyblue',color='none',s=80)
plt.scatter(noi_p_arr_19[regnum,i,:],noi_e_arr_19[regnum,i,:],marker=psyms[i],color='deepskyblue',label=modnames[i])
# points
plt.scatter(irr_p_arr_19[regnum,i,0],irr_e_arr_19[regnum,i,0],marker='s',edgecolors='maroon',color='none',s=80)
plt.scatter(irr_p_arr_19[regnum,i,-1],irr_e_arr_19[regnum,i,-1],marker='o',edgecolors='maroon',color='none',s=80)
plt.scatter(irr_p_arr_19[regnum,i,:],irr_e_arr_19[regnum,i,:],marker=psyms[i],color='maroon')

plt.xlim((320.,720.))
plt.ylim((275.,710.))

plt.xlabel(r'$P$ [mm yr$^{-1}$]')
#plt.ylabel(r'$ET$ [mm yr$^{-1}$]')
plt.title(regnamesfull[regnum])
#plt.legend()

# Nature-style annotations

plt.annotate('b',(50.,320.),xycoords='figure
points',fontweight='bold',fontsize='x-large')
plt.annotate('c',(240.,320.),xycoords='figure
points',fontweight='bold',fontsize='x-large')


plt.savefig('irrmip_p_and_e.pdf')



