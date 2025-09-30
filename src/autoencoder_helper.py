import os, sys, imp
sys.path.append('/home/seyedkazemi/codes/mskhelper/')
import pandas as pd, numpy as np
import datetime, pytz
import data_provider, mohsenutils, m100_preprocessing_helper, flag_helper
import imp 
from  matplotlib import pyplot as plt
import torch 
import torch.nn as nn
import torch.functional as f
from torch.utils.data import DataLoader, Dataset
# from torchsummary import summary
from torchinfo import summary
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from contextlib import redirect_stdout
import json
import datetime, pytz
imp.reload(data_provider)
imp.reload(mohsenutils)
imp.reload(m100_preprocessing_helper)
imp.reload(flag_helper)


################################
#         Data Provider        #
################################


# def flags_data(rack='205'):
#     cols=['r'+str(rack)+'n'+str(chassis).zfill(2) for chassis in range(1,21)]
#     cols.append('timestamp')
#     # print(cols)
#     path = '/home/seyedkazemi/dataset/M100_Complete/'

#     kwargs_cols = {'parse_dates':True, 'date_parser':pd.to_datetime, 'index_col':'timestamp', 'usecols':cols}
#     kwargs = {'parse_dates':True, 'date_parser':pd.to_datetime, 'index_col':'timestamp'}


#     ambient_dt = pd.read_csv(path+'ambient_big.csv.gz', **kwargs_cols)
#     ambient_dt.columns += '_ambient'
    
#     pcie_dt = pd.read_csv(path+'pcie_big.csv.gz',**kwargs_cols)
#     pcie_dt.columns += '_pcie_dt'
    
#     fsnd_dt = pd.read_csv(path+'fan_nodes_big.csv.gz',**kwargs_cols)
#     fsnd_dt.columns += '_fsnd_dt'
    
#     psnd_dt = pd.read_csv(path+'ps_big.csv.gz',**kwargs_cols)
#     psnd_dt.columns += '_psnd_dt'
    
#     gpu0_dt = pd.read_csv(path+'gpu0_core_temp_big.csv.gz', **kwargs_cols)
#     gpu0_dt.columns += '_gpu0_dt'
#     gpu1_dt = pd.read_csv(path+'gpu1_core_temp_big.csv.gz', **kwargs_cols)
#     gpu1_dt.columns += '_gpu1_dt'
#     gpu3_dt = pd.read_csv(path+'gpu3_core_temp_big.csv.gz', **kwargs_cols)
#     gpu3_dt.columns += '_gpu3_dt'
#     gpu4_dt = pd.read_csv(path+'gpu4_core_temp_big.csv.gz', **kwargs_cols)
#     gpu4_dt.columns += '_gpu4_dt'
    
    
#     gpu0_dt, gpu1_dt, gpu3_dt, gpu4_dt = mohsenutils.same_index_range(gpu0_dt, gpu1_dt, gpu3_dt, gpu4_dt)
# #     gpu_dt = mohsenutils.mrg(None,'outer',gpu0_dt, gpu1_dt, gpu3_dt, gpu4_dt)

#     cpu0_dt = pd.read_csv(path+'cores_temp0_big.csv.gz', **kwargs_cols)
#     cpu0_dt.columns += '_cpu0_dt'
#     cpu1_dt = pd.read_csv(path+'cores_temp1_big.csv.gz', **kwargs_cols)
#     cpu1_dt.columns += '_cpu1_dt'
#     cpu0_dt, cpu1_dt = mohsenutils.same_index_range(cpu0_dt, cpu1_dt)
# #     cpu_dt = mohsenutils.mrg(None,'outer',cpu0_dt, cpu1_dt)

#     Tot_ict_dt = pd.read_csv(path+'Tot_ict_big.csv.gz',**kwargs)

#     for df in [ambient_dt, pcie_dt, gpu0_dt, gpu1_dt, gpu3_dt, gpu4_dt, cpu0_dt, cpu1_dt, fsnd_dt, psnd_dt]:
#         assert df.shape[1]==20, 'The number of selected nodes is NOT correct it should be 20. Please check the rack name.'



#     Tot_qpompe_dt = pd.read_csv(path+'Tot_qpompe_big.csv.gz',**kwargs)
#     RDHX_dt = flag_helper.rdhx(pd.read_csv(path+'RDHX_big.csv.gz',**kwargs))
#     Tot_chiller_dt = pd.read_csv(path+'Tot_chiller_big.csv.gz',**kwargs)

#     Supply_Air_Temperature_dt = pd.read_csv(path+'Supply_Air_Temperature_big.csv.gz',**kwargs)
#     Supply_Air_Temperature_dt.columns += '_SAT'
    
#     Return_Air_Temperature_dt = pd.read_csv(path+'Return_Air_Temperature_big.csv.gz',**kwargs)
#     Return_Air_Temperature_dt.columns += '_RAT'
    
#     Free_Cooling_Valve_Open_Position_dt = pd.read_csv(path+'Free_Cooling_Valve_Open_Position_big.csv.gz',**kwargs)[['cdz1','cdz2','cdz3','cdz4']]
#     Free_Cooling_Valve_Open_Position_dt.columns += '_FCVOP'
    
#     Tot_cdz_dt = pd.read_csv(path+'Tot_cdz_big.csv.gz',**kwargs)
    
#     FScrac_dt = pd.read_csv(path+'Fan_Speed_big.csv.gz',**kwargs)
#     FScrac_dt.columns += '_FSCRAC'
    
#     Free_Cooling_Status_dt = pd.read_csv(path+'Free_Cooling_Status_big.csv.gz',**kwargs)[['cdz1','cdz2','cdz3','cdz4']]
#     Free_Cooling_Status_dt.columns += '_FCS'
    
#     Compressor_Utilization_dt = pd.read_csv(path+'Compressor_Utilization_big.csv.gz',**kwargs)
#     Compressor_Utilization_dt.columns += '_CU'

#     temp_dt = pd.read_csv(path+'temp_big.csv.gz', **kwargs)[['current']]


#     # ambient_dt,pcie_dt,fsnd_dt,psnd_dt,
#     # gpu0_dt, gpu1_dt, gpu3_dt, gpu4_dt,
#     # cpu0_dt, cpu1_dt,Tot_ict_dt,Tot_qpompe_dt,
#     # RDHX_dt,Tot_chiller_dt,Supply_Air_Temperature_dt,
#     # Return_Air_Temperature_dt,Free_Cooling_Valve_Open_Position_dt,
#     # Tot_cdz_dt,FScrac_dt,Free_Cooling_Status_dt,Compressor_Utilization_dt,temp_dt


#     flags99 = pd.read_csv('/home/seyedkazemi/dataset/Marconi100-Cooked/flags9901.csv',**kwargs)
#     flags = pd.DataFrame(flags99.astype(bool).astype(int).sum(axis=1), columns=['sum_flags'])
#     # display(flags)


#     flags, ambient_dt = mohsenutils.same_index_range(flags, ambient_dt)




#     data = mohsenutils.mrg(None, 'inner', flags, 
#                            Tot_ict_dt, Tot_qpompe_dt, Tot_cdz_dt, Tot_chiller_dt, psnd_dt,
#                            ambient_dt, Supply_Air_Temperature_dt,Return_Air_Temperature_dt, gpu0_dt, gpu1_dt, gpu3_dt, gpu4_dt, cpu0_dt, cpu1_dt,temp_dt,
#                            RDHX_dt,Compressor_Utilization_dt, pcie_dt,fsnd_dt,Free_Cooling_Valve_Open_Position_dt,FScrac_dt,Free_Cooling_Status_dt)
#     print(f'Shape of data {data.shape}')
#     # Drop rows with Null cell. 
#     data = data.drop(data[data.isnull().any(axis=1)].index)
#     print(f'Shape of data after drop null data {data.shape}')
    
#     return data
    

    
def flags_data(rack='205', 
               ambient_dt=True,
               pcie_dt=True,
               fsnd_dt=True,
               psnd_dt=True,
               gpu0_dt=True,
               gpu1_dt=True,
               gpu3_dt=True,
               gpu4_dt=True,
               cpu0_dt=True,
               cpu1_dt=True,
               Tot_ict_dt=True,
               Tot_qpompe_dt=True,
               RDHX_dt=True,
               Tot_chiller_dt=True,
               Supply_Air_Temperature_dt=True,
               Return_Air_Temperature_dt=True,
               Free_Cooling_Valve_Open_Position_dt=True,
               Tot_cdz_dt=True,
               FScrac_dt=True,
               Free_Cooling_Status_dt=True,
               Compressor_Utilization_dt=True,
               temp_dt=True):
    
    lst_dt = []
    cols=['r'+str(rack)+'n'+str(chassis).zfill(2) for chassis in range(1,21)]
    cols.append('timestamp')
    # print(cols)
    path = '/home/seyedkazemi/dataset/M100_Study/'

    kwargs_cols = {'parse_dates':True, 'date_parser':pd.to_datetime, 'index_col':'timestamp', 'usecols':cols}
    kwargs = {'parse_dates':True, 'date_parser':pd.to_datetime, 'index_col':'timestamp'}


    if ambient_dt==True:
        ambient_dt = pd.read_csv(path+'ambient_big.csv.gz', **kwargs_cols)
        ambient_dt.columns += '_ambient'
        lst_dt.append(ambient_dt)
        
    if pcie_dt==True:
        pcie_dt = pd.read_csv(path+'pcie_big.csv.gz',**kwargs_cols)
        pcie_dt.columns += '_pcie_dt'
        lst_dt.append(pcie_dt)
    
    if fsnd_dt==True:
        fsnd_dt = pd.read_csv(path+'fan_nodes_big.csv.gz',**kwargs_cols)
        fsnd_dt.columns += '_fsnd_dt'
        lst_dt.append(fsnd_dt)
    
    if psnd_dt==True:
        psnd_dt = pd.read_csv(path+'ps_big.csv.gz',**kwargs_cols)
        psnd_dt.columns += '_psnd_dt'
        lst_dt.append(psnd_dt)
    
    if gpu0_dt==True:
        gpu0_dt = pd.read_csv(path+'gpu0_core_temp_big.csv.gz', **kwargs_cols)
        gpu0_dt.columns += '_gpu0_dt'
        lst_dt.append(gpu0_dt)
    if gpu1_dt==True:
        gpu1_dt = pd.read_csv(path+'gpu1_core_temp_big.csv.gz', **kwargs_cols)
        gpu1_dt.columns += '_gpu1_dt'
        lst_dt.append(gpu1_dt)
    if gpu3_dt==True:
        gpu3_dt = pd.read_csv(path+'gpu3_core_temp_big.csv.gz', **kwargs_cols)
        gpu3_dt.columns += '_gpu3_dt'
        lst_dt.append(gpu3_dt)
    if gpu4_dt==True:
        gpu4_dt = pd.read_csv(path+'gpu4_core_temp_big.csv.gz', **kwargs_cols)
        gpu4_dt.columns += '_gpu4_dt'
        lst_dt.append(gpu4_dt)
    


    if cpu0_dt==True:
        cpu0_dt = pd.read_csv(path+'cores_temp0_big.csv.gz', **kwargs_cols)
        cpu0_dt.columns += '_cpu0_dt'
        lst_dt.append(cpu0_dt)
    if cpu1_dt==True:
        cpu1_dt = pd.read_csv(path+'cores_temp1_big.csv.gz', **kwargs_cols)
        cpu1_dt.columns += '_cpu1_dt'
        lst_dt.append(cpu1_dt)
        
    for df in lst_dt:
        assert df.shape[1]==20, 'The number of selected nodes is NOT correct it should be 20. Please check the rack name.'

    if Tot_ict_dt==True:
        Tot_ict_dt = pd.read_csv(path+'Tot_ict_big.csv.gz',**kwargs)
        lst_dt.append(Tot_ict_dt)


#     gpu0_dt, gpu1_dt, gpu3_dt, gpu4_dt = mohsenutils.same_index_range(gpu0_dt, gpu1_dt, gpu3_dt, gpu4_dt)
#     gpu_dt = mohsenutils.mrg(None,'outer',gpu0_dt, gpu1_dt, gpu3_dt, gpu4_dt)
#     cpu0_dt, cpu1_dt = mohsenutils.same_index_range(cpu0_dt, cpu1_dt)
#     cpu_dt = mohsenutils.mrg(None,'outer',cpu0_dt, cpu1_dt)
        

    if Tot_qpompe_dt==True:
        Tot_qpompe_dt = pd.read_csv(path+'Tot_qpompe_big.csv.gz',**kwargs)
        lst_dt.append(Tot_qpompe_dt)
        
    if RDHX_dt==True:
        RDHX_dt = flag_helper.rdhx(pd.read_csv(path+'RDHX_big.csv.gz',**kwargs))
        lst_dt.append(RDHX_dt)
        
    if Tot_chiller_dt==True:
        Tot_chiller_dt = pd.read_csv(path+'Tot_chiller_big.csv.gz',**kwargs)
        lst_dt.append(Tot_chiller_dt)

    if Supply_Air_Temperature_dt==True:
        Supply_Air_Temperature_dt = pd.read_csv(path+'Supply_Air_Temperature_big.csv.gz',**kwargs)
        Supply_Air_Temperature_dt.columns += '_SAT'
        lst_dt.append(Supply_Air_Temperature_dt)
    
    if Return_Air_Temperature_dt==True:
        Return_Air_Temperature_dt = pd.read_csv(path+'Return_Air_Temperature_big.csv.gz',**kwargs)
        Return_Air_Temperature_dt.columns += '_RAT'
        lst_dt.append(Return_Air_Temperature_dt)
    
    if Free_Cooling_Valve_Open_Position_dt==True:
        Free_Cooling_Valve_Open_Position_dt = pd.read_csv(path+'Free_Cooling_Valve_Open_Position_big.csv.gz',**kwargs)[['cdz1','cdz2','cdz3','cdz4']]
        Free_Cooling_Valve_Open_Position_dt.columns += '_FCVOP'
        lst_dt.append(Free_Cooling_Valve_Open_Position_dt)

    if Tot_cdz_dt==True:
        Tot_cdz_dt = pd.read_csv(path+'Tot_cdz_big.csv.gz',**kwargs)
        lst_dt.append(Tot_cdz_dt)
    
    if FScrac_dt==True:
        FScrac_dt = pd.read_csv(path+'Fan_Speed_big.csv.gz',**kwargs)
        FScrac_dt.columns += '_FSCRAC'
        lst_dt.append(FScrac_dt)
    
    if Free_Cooling_Status_dt==True:
        Free_Cooling_Status_dt = pd.read_csv(path+'Free_Cooling_Status_big.csv.gz',**kwargs)[['cdz1','cdz2','cdz3','cdz4']]
        Free_Cooling_Status_dt.columns += '_FCS'
        lst_dt.append(Free_Cooling_Status_dt)
    
    if Compressor_Utilization_dt==True:
        Compressor_Utilization_dt = pd.read_csv(path+'Compressor_Utilization_big.csv.gz',**kwargs)
        Compressor_Utilization_dt.columns += '_CU'
        lst_dt.append(Compressor_Utilization_dt)

    if temp_dt==True:
        temp_dt = pd.read_csv(path+'temp_big.csv.gz', **kwargs)[['current']]
        lst_dt.append(temp_dt)




    flags99 = pd.read_csv('/home/seyedkazemi/dataset/Marconi100-Cooked/flags9901.csv',**kwargs)
    flags = pd.DataFrame(flags99.astype(bool).astype(int).sum(axis=1), columns=['sum_flags'])
    # display(flags)
    lst_dt = mohsenutils.same_index_range(flags, *lst_dt)    
    data = mohsenutils.mrg(None, 'inner', *lst_dt)

    
    
    # ambient_dt,pcie_dt,fsnd_dt,psnd_dt,
    # gpu0_dt, gpu1_dt, gpu3_dt, gpu4_dt,
    # cpu0_dt, cpu1_dt,Tot_ict_dt,Tot_qpompe_dt,
    # RDHX_dt,Tot_chiller_dt,Supply_Air_Temperature_dt,
    # Return_Air_Temperature_dt,Free_Cooling_Valve_Open_Position_dt,
    # Tot_cdz_dt,FScrac_dt,Free_Cooling_Status_dt,Compressor_Utilization_dt,temp_dt
    
#     data = mohsenutils.mrg(None, 'inner', flags, 
#                            Tot_ict_dt, Tot_qpompe_dt, Tot_cdz_dt, Tot_chiller_dt, psnd_dt,
#                            ambient_dt, Supply_Air_Temperature_dt,Return_Air_Temperature_dt, gpu0_dt, gpu1_dt, gpu3_dt, gpu4_dt, cpu0_dt, cpu1_dt,temp_dt,
#                            RDHX_dt,Compressor_Utilization_dt, pcie_dt,fsnd_dt,Free_Cooling_Valve_Open_Position_dt,FScrac_dt,Free_Cooling_Status_dt)
    
    
    
    
#     print(f'Shape of data {data.shape}')
    
#     # Drop rows with Null cell. 
#     data = data.drop(data[data.isnull().any(axis=1)].index)
#     print(f'Shape of data after drop null data {data.shape}')
    
    return data
    
    
    

    
    
    
    

    
def flags_data_new(rack='205', 
               ambient_dt=True,
               pcie_dt=True,
               fsnd_dt=True,
               psnd_dt=True,
               gpu0_dt=True,
               gpu1_dt=True,
               gpu3_dt=True,
               gpu4_dt=True,
               cpu0_dt=True,
               cpu1_dt=True,
               Tot_ict_dt=True,
               Tot_qpompe_dt=True,
               RDHX_dt=True,
               Tot_chiller_dt=True,
               Supply_Air_Temperature_dt=True,
               Return_Air_Temperature_dt=True,
               Free_Cooling_Valve_Open_Position_dt=True,
               Tot_cdz_dt=True,
               FScrac_dt=True,
               Free_Cooling_Status_dt=True,
               Compressor_Utilization_dt=True,
               temp_dt=True):
    
    lst_dt = []
    cols=['r'+str(rack)+'n'+str(chassis).zfill(2) for chassis in range(1,21)]
    cols.append('timestamp')
    # print(cols)
    path = '/home/seyedkazemi/dataset/M100_Complete/'

    kwargs_cols = {'parse_dates':True, 'date_parser':pd.to_datetime, 'index_col':'timestamp', 'usecols':cols}
    kwargs = {'parse_dates':True, 'date_parser':pd.to_datetime, 'index_col':'timestamp'}


    if ambient_dt==True:
        ambient_dt = pd.read_csv(path+'ambient_big.csv.gz', **kwargs_cols)
        ambient_dt.columns += '_ambient'
        lst_dt.append(ambient_dt)
        
    if pcie_dt==True:
        pcie_dt = pd.read_csv(path+'pcie_big.csv.gz',**kwargs_cols)
        pcie_dt.columns += '_pcie_dt'
        lst_dt.append(pcie_dt)
    
    if fsnd_dt==True:
        fsnd_dt = pd.read_csv(path+'fan_nodes_big.csv.gz',**kwargs_cols)
        fsnd_dt.columns += '_fsnd_dt'
        lst_dt.append(fsnd_dt)
    
    if psnd_dt==True:
        psnd_dt = pd.read_csv(path+'ps_big.csv.gz',**kwargs_cols)
        psnd_dt.columns += '_psnd_dt'
        lst_dt.append(psnd_dt)
    
    if gpu0_dt==True:
        gpu0_dt = pd.read_csv(path+'gpu0_core_temp_big.csv.gz', **kwargs_cols)
        gpu0_dt.columns += '_gpu0_dt'
        lst_dt.append(gpu0_dt)
    if gpu1_dt==True:
        gpu1_dt = pd.read_csv(path+'gpu1_core_temp_big.csv.gz', **kwargs_cols)
        gpu1_dt.columns += '_gpu1_dt'
        lst_dt.append(gpu1_dt)
    if gpu3_dt==True:
        gpu3_dt = pd.read_csv(path+'gpu3_core_temp_big.csv.gz', **kwargs_cols)
        gpu3_dt.columns += '_gpu3_dt'
        lst_dt.append(gpu3_dt)
    if gpu4_dt==True:
        gpu4_dt = pd.read_csv(path+'gpu4_core_temp_big.csv.gz', **kwargs_cols)
        gpu4_dt.columns += '_gpu4_dt'
        lst_dt.append(gpu4_dt)
    


    if cpu0_dt==True:
        cpu0_dt = pd.read_csv(path+'cores_temp0_big.csv.gz', **kwargs_cols)
        cpu0_dt.columns += '_cpu0_dt'
        lst_dt.append(cpu0_dt)
    if cpu1_dt==True:
        cpu1_dt = pd.read_csv(path+'cores_temp1_big.csv.gz', **kwargs_cols)
        cpu1_dt.columns += '_cpu1_dt'
        lst_dt.append(cpu1_dt)
        
    for df in lst_dt:
        assert df.shape[1]==20, 'The number of selected nodes is NOT correct it should be 20. Please check the rack name.'

    if Tot_ict_dt==True:
        Tot_ict_dt = pd.read_csv(path+'Tot_ict_big.csv.gz',**kwargs)
        lst_dt.append(Tot_ict_dt)


#     gpu0_dt, gpu1_dt, gpu3_dt, gpu4_dt = mohsenutils.same_index_range(gpu0_dt, gpu1_dt, gpu3_dt, gpu4_dt)
#     gpu_dt = mohsenutils.mrg(None,'outer',gpu0_dt, gpu1_dt, gpu3_dt, gpu4_dt)
#     cpu0_dt, cpu1_dt = mohsenutils.same_index_range(cpu0_dt, cpu1_dt)
#     cpu_dt = mohsenutils.mrg(None,'outer',cpu0_dt, cpu1_dt)
        

    if Tot_qpompe_dt==True:
        Tot_qpompe_dt = pd.read_csv(path+'Tot_qpompe_big.csv.gz',**kwargs)
        lst_dt.append(Tot_qpompe_dt)
        
    if RDHX_dt==True:
        RDHX_dt = flag_helper.rdhx(pd.read_csv(path+'RDHX_big.csv.gz',**kwargs))
        lst_dt.append(RDHX_dt)
        
    if Tot_chiller_dt==True:
        Tot_chiller_dt = pd.read_csv(path+'Tot_chiller_big.csv.gz',**kwargs)
        lst_dt.append(Tot_chiller_dt)

    if Supply_Air_Temperature_dt==True:
        Supply_Air_Temperature_dt = pd.read_csv(path+'Supply_Air_Temperature_big.csv.gz',**kwargs)
        Supply_Air_Temperature_dt.columns += '_SAT'
        lst_dt.append(Supply_Air_Temperature_dt)
    
    if Return_Air_Temperature_dt==True:
        Return_Air_Temperature_dt = pd.read_csv(path+'Return_Air_Temperature_big.csv.gz',**kwargs)
        Return_Air_Temperature_dt.columns += '_RAT'
        lst_dt.append(Return_Air_Temperature_dt)
    
    if Free_Cooling_Valve_Open_Position_dt==True:
        Free_Cooling_Valve_Open_Position_dt = pd.read_csv(path+'Free_Cooling_Valve_Open_Position_big.csv.gz',**kwargs)[['cdz1','cdz2','cdz3','cdz4']]
        Free_Cooling_Valve_Open_Position_dt.columns += '_FCVOP'
        lst_dt.append(Free_Cooling_Valve_Open_Position_dt)

    if Tot_cdz_dt==True:
        Tot_cdz_dt = pd.read_csv(path+'Tot_cdz_big.csv.gz',**kwargs)
        lst_dt.append(Tot_cdz_dt)
    
    if FScrac_dt==True:
        FScrac_dt = pd.read_csv(path+'Fan_Speed_big.csv.gz',**kwargs)
        FScrac_dt.columns += '_FSCRAC'
        lst_dt.append(FScrac_dt)
    
    if Free_Cooling_Status_dt==True:
        Free_Cooling_Status_dt = pd.read_csv(path+'Free_Cooling_Status_big.csv.gz',**kwargs)[['cdz1','cdz2','cdz3','cdz4']]
        Free_Cooling_Status_dt.columns += '_FCS'
        lst_dt.append(Free_Cooling_Status_dt)
    
    if Compressor_Utilization_dt==True:
        Compressor_Utilization_dt = pd.read_csv(path+'Compressor_Utilization_big.csv.gz',**kwargs)
        Compressor_Utilization_dt.columns += '_CU'
        lst_dt.append(Compressor_Utilization_dt)

    if temp_dt==True:
        temp_dt = pd.read_csv(path+'temp_big.csv.gz', **kwargs)[['current']]
        lst_dt.append(temp_dt)





#     flags99 = pd.read_csv('/home/seyedkazemi/dataset/Marconi100-Cooked/new-flags9901.csv',**kwargs)
# 2021_08_01_23_50 actually the it is 2021-08-17 23:50:00 and I think I use the quantile of data until 2021_08_01_23_50 to compute the thresholds. 
    flags99 = pd.read_csv('/home/seyedkazemi/dataset/Marconi100-Cooked/flags9901_2021_08_01_23_50.csv',**kwargs)
    flags = pd.DataFrame(flags99.astype(bool).astype(int).sum(axis=1), columns=['sum_flags'])
    # display(flags)


    lst_dt = mohsenutils.same_index_range(flags, *lst_dt)    
    data = mohsenutils.mrg(None,'inner', *lst_dt, suffixe_lst=None)
#     on, how,  df, suffixe_lst=None, *dataframe

    
    
    # ambient_dt,pcie_dt,fsnd_dt,psnd_dt,
    # gpu0_dt, gpu1_dt, gpu3_dt, gpu4_dt,
    # cpu0_dt, cpu1_dt,Tot_ict_dt,Tot_qpompe_dt,
    # RDHX_dt,Tot_chiller_dt,Supply_Air_Temperature_dt,
    # Return_Air_Temperature_dt,Free_Cooling_Valve_Open_Position_dt,
    # Tot_cdz_dt,FScrac_dt,Free_Cooling_Status_dt,Compressor_Utilization_dt,temp_dt
    
#     data = mohsenutils.mrg(None, 'inner', flags, 
#                            Tot_ict_dt, Tot_qpompe_dt, Tot_cdz_dt, Tot_chiller_dt, psnd_dt,
#                            ambient_dt, Supply_Air_Temperature_dt,Return_Air_Temperature_dt, gpu0_dt, gpu1_dt, gpu3_dt, gpu4_dt, cpu0_dt, cpu1_dt,temp_dt,
#                            RDHX_dt,Compressor_Utilization_dt, pcie_dt,fsnd_dt,Free_Cooling_Valve_Open_Position_dt,FScrac_dt,Free_Cooling_Status_dt)
    
    
    
    
#     print(f'Shape of data {data.shape}')
    
#     # Drop rows with Null cell. 
#     data = data.drop(data[data.isnull().any(axis=1)].index)
#     print(f'Shape of data after drop null data {data.shape}')
    
    return data

    
    
    
def flags_data_2022_04_07(rack='205', 
               ambient_dt=True,
               pcie_dt=True,
               fsnd_dt=True,
               psnd_dt=True,
               gpu0_dt=True,
               gpu1_dt=True,
               gpu3_dt=True,
               gpu4_dt=True,
               cpu0_dt=True,
               cpu1_dt=True,
               Tot_ict_dt=True,
               Tot_qpompe_dt=True,
               RDHX_dt=True,
               Tot_chiller_dt=True,
               Supply_Air_Temperature_dt=True,
               Return_Air_Temperature_dt=True,
               Free_Cooling_Valve_Open_Position_dt=True,
               Tot_cdz_dt=True,
               FScrac_dt=True,
               Free_Cooling_Status_dt=True,
               Compressor_Utilization_dt=True,
               temp_dt=True):
    
    lst_dt = []
    cols=['r'+str(rack)+'n'+str(chassis).zfill(2) for chassis in range(1,21)]
    cols.append('timestamp')
    # print(cols)
    path = '/home/seyedkazemi/dataset/M100_Complete/'

    kwargs_cols = {'parse_dates':True, 'date_parser':pd.to_datetime, 'index_col':'timestamp', 'usecols':cols}
    kwargs = {'parse_dates':True, 'date_parser':pd.to_datetime, 'index_col':'timestamp'}


    if ambient_dt==True:
        ambient_dt = pd.read_csv(path+'ambient_big.csv.gz', **kwargs_cols)
        ambient_dt.columns += '_ambient'
        lst_dt.append(ambient_dt)
        
    if pcie_dt==True:
        pcie_dt = pd.read_csv(path+'pcie_big.csv.gz',**kwargs_cols)
        pcie_dt.columns += '_pcie_dt'
        lst_dt.append(pcie_dt)
    
    if fsnd_dt==True:
        fsnd_dt = pd.read_csv(path+'fan_nodes_big.csv.gz',**kwargs_cols)
        fsnd_dt.columns += '_fsnd_dt'
        lst_dt.append(fsnd_dt)
    
    if psnd_dt==True:
        psnd_dt = pd.read_csv(path+'ps_big.csv.gz',**kwargs_cols)
        psnd_dt.columns += '_psnd_dt'
        lst_dt.append(psnd_dt)
    
    if gpu0_dt==True:
        gpu0_dt = pd.read_csv(path+'gpu0_core_temp_big.csv.gz', **kwargs_cols)
        gpu0_dt.columns += '_gpu0_dt'
        lst_dt.append(gpu0_dt)
    if gpu1_dt==True:
        gpu1_dt = pd.read_csv(path+'gpu1_core_temp_big.csv.gz', **kwargs_cols)
        gpu1_dt.columns += '_gpu1_dt'
        lst_dt.append(gpu1_dt)
    if gpu3_dt==True:
        gpu3_dt = pd.read_csv(path+'gpu3_core_temp_big.csv.gz', **kwargs_cols)
        gpu3_dt.columns += '_gpu3_dt'
        lst_dt.append(gpu3_dt)
    if gpu4_dt==True:
        gpu4_dt = pd.read_csv(path+'gpu4_core_temp_big.csv.gz', **kwargs_cols)
        gpu4_dt.columns += '_gpu4_dt'
        lst_dt.append(gpu4_dt)
    


    if cpu0_dt==True:
        cpu0_dt = pd.read_csv(path+'cores_temp0_big.csv.gz', **kwargs_cols)
        cpu0_dt.columns += '_cpu0_dt'
        lst_dt.append(cpu0_dt)
    if cpu1_dt==True:
        cpu1_dt = pd.read_csv(path+'cores_temp1_big.csv.gz', **kwargs_cols)
        cpu1_dt.columns += '_cpu1_dt'
        lst_dt.append(cpu1_dt)
        
    for df in lst_dt:
        assert df.shape[1]==20, 'The number of selected nodes is NOT correct it should be 20. Please check the rack name.'

    if Tot_ict_dt==True:
        Tot_ict_dt = pd.read_csv(path+'Tot_ict_big.csv.gz',**kwargs)
        lst_dt.append(Tot_ict_dt)


#     gpu0_dt, gpu1_dt, gpu3_dt, gpu4_dt = mohsenutils.same_index_range(gpu0_dt, gpu1_dt, gpu3_dt, gpu4_dt)
#     gpu_dt = mohsenutils.mrg(None,'outer',gpu0_dt, gpu1_dt, gpu3_dt, gpu4_dt)
#     cpu0_dt, cpu1_dt = mohsenutils.same_index_range(cpu0_dt, cpu1_dt)
#     cpu_dt = mohsenutils.mrg(None,'outer',cpu0_dt, cpu1_dt)
        

    if Tot_qpompe_dt==True:
        Tot_qpompe_dt = pd.read_csv(path+'Tot_qpompe_big.csv.gz',**kwargs)
        lst_dt.append(Tot_qpompe_dt)
        
    if RDHX_dt==True:
        RDHX_dt = flag_helper.rdhx(pd.read_csv(path+'RDHX_big.csv.gz',**kwargs))
        lst_dt.append(RDHX_dt)
        
    if Tot_chiller_dt==True:
        Tot_chiller_dt = pd.read_csv(path+'Tot_chiller_big.csv.gz',**kwargs)
        lst_dt.append(Tot_chiller_dt)

    if Supply_Air_Temperature_dt==True:
        Supply_Air_Temperature_dt = pd.read_csv(path+'Supply_Air_Temperature_big.csv.gz',**kwargs)
        Supply_Air_Temperature_dt.columns += '_SAT'
        lst_dt.append(Supply_Air_Temperature_dt)
    
    if Return_Air_Temperature_dt==True:
        Return_Air_Temperature_dt = pd.read_csv(path+'Return_Air_Temperature_big.csv.gz',**kwargs)
        Return_Air_Temperature_dt.columns += '_RAT'
        lst_dt.append(Return_Air_Temperature_dt)
    
    if Free_Cooling_Valve_Open_Position_dt==True:
        Free_Cooling_Valve_Open_Position_dt = pd.read_csv(path+'Free_Cooling_Valve_Open_Position_big.csv.gz',**kwargs)[['cdz1','cdz2','cdz3','cdz4']]
        Free_Cooling_Valve_Open_Position_dt.columns += '_FCVOP'
        lst_dt.append(Free_Cooling_Valve_Open_Position_dt)

    if Tot_cdz_dt==True:
        Tot_cdz_dt = pd.read_csv(path+'Tot_cdz_big.csv.gz',**kwargs)
        lst_dt.append(Tot_cdz_dt)
    
    if FScrac_dt==True:
        FScrac_dt = pd.read_csv(path+'Fan_Speed_big.csv.gz',**kwargs)
        FScrac_dt.columns += '_FSCRAC'
        lst_dt.append(FScrac_dt)
    
    if Free_Cooling_Status_dt==True:
        Free_Cooling_Status_dt = pd.read_csv(path+'Free_Cooling_Status_big.csv.gz',**kwargs)[['cdz1','cdz2','cdz3','cdz4']]
        Free_Cooling_Status_dt.columns += '_FCS'
        lst_dt.append(Free_Cooling_Status_dt)
    
    if Compressor_Utilization_dt==True:
        Compressor_Utilization_dt = pd.read_csv(path+'Compressor_Utilization_big.csv.gz',**kwargs)
        Compressor_Utilization_dt.columns += '_CU'
        lst_dt.append(Compressor_Utilization_dt)

    if temp_dt==True:
        temp_dt = pd.read_csv(path+'temp_big.csv.gz', **kwargs)[['current']]
        lst_dt.append(temp_dt)





# #     flags99 = pd.read_csv('/home/seyedkazemi/dataset/Marconi100-Cooked/new-flags9901.csv',**kwargs)
# # 2021_08_01_23_50 actually the it is 2021-08-17 23:50:00 and I think I use the quantile of data until 2021_08_01_23_50 to compute the thresholds. 
#     flags99 = pd.read_csv('/home/seyedkazemi/dataset/Marconi100-Cooked/flags9901_2021_08_01_23_50.csv',**kwargs)
#     flags = pd.DataFrame(flags99.astype(bool).astype(int).sum(axis=1), columns=['sum_flags'])
#     # display(flags)


    lst_dt = mohsenutils.same_index_range(*lst_dt)    
    data = mohsenutils.mrg(None,'inner', *lst_dt, suffixe_lst=None)
#     on, how,  df, suffixe_lst=None, *dataframe

    
    
    # ambient_dt,pcie_dt,fsnd_dt,psnd_dt,
    # gpu0_dt, gpu1_dt, gpu3_dt, gpu4_dt,
    # cpu0_dt, cpu1_dt,Tot_ict_dt,Tot_qpompe_dt,
    # RDHX_dt,Tot_chiller_dt,Supply_Air_Temperature_dt,
    # Return_Air_Temperature_dt,Free_Cooling_Valve_Open_Position_dt,
    # Tot_cdz_dt,FScrac_dt,Free_Cooling_Status_dt,Compressor_Utilization_dt,temp_dt
    
#     data = mohsenutils.mrg(None, 'inner', flags, 
#                            Tot_ict_dt, Tot_qpompe_dt, Tot_cdz_dt, Tot_chiller_dt, psnd_dt,
#                            ambient_dt, Supply_Air_Temperature_dt,Return_Air_Temperature_dt, gpu0_dt, gpu1_dt, gpu3_dt, gpu4_dt, cpu0_dt, cpu1_dt,temp_dt,
#                            RDHX_dt,Compressor_Utilization_dt, pcie_dt,fsnd_dt,Free_Cooling_Valve_Open_Position_dt,FScrac_dt,Free_Cooling_Status_dt)
    
    
    
    
#     print(f'Shape of data {data.shape}')
    
#     # Drop rows with Null cell. 
#     data = data.drop(data[data.isnull().any(axis=1)].index)
#     print(f'Shape of data after drop null data {data.shape}')
    
    return data
    
    
    
    
    
#########################################################
#                          MLP                          #
#########################################################

################################
#        MLP  DataLoader       #
################################


class MLP_AE_dataset(Dataset):
    def __init__(self,dataframe):
        self.dataframe = dataframe
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        return self.dataframe[idx,:], self.dataframe[idx,:]

    
def MLP_AE_data_provider(dataframe, test_size, grey_data_train_threshold, batch_size_train, batch_size_test, shuffle):
    """
        
    return {'abnormal_data_index':abnormal_data_index,
            'normal_data_train_index':normal_data_train_index,
            'normal_data_test_index':normal_data_test_index,
            'grey_data_index':grey_data_index,
            'grey_data_train_index':grey_data_train_index,
            'train_data_index':train_data_index,
            'fit_scaler_fun':scaler,
            'train_dataloader':train_dataloader,
            'test_dataloader':test_dataloader,
            'abnormal_dataloader':abnormal_dataloader}
            
    
    """
    print(f'Shape of dataframe {dataframe.shape}')
    # Drop rows with Null cell. 
    dataframe = dataframe.drop(dataframe[dataframe.isnull().any(axis=1)].index)
    print(f'Shape of dataframe after drop null data {dataframe.shape}')

    normal_data   = dataframe.query(  "sum_flags==0")
    abnormal_data = dataframe.query(  "sum_flags>25")
    grey_data =     dataframe.query("0<sum_flags<=25")
    grey_data_train=dataframe.query("0<sum_flags<="+str(grey_data_train_threshold))


# Plots
    display('Normal Dataset Sum of Falgs')
    normal_data[  ['sum_flags']].hist()
    plt.show()
    display('Abnormal Dataset Sum of Falgs')
    abnormal_data[['sum_flags']].hist()
    plt.show()
    display('Grey Dataset Sum of Falgs')
    grey_data[['sum_flags']].hist()
    plt.show()
    display('Grey Train Dataset Sum of Falgs')
    grey_data_train[['sum_flags']].hist()
    plt.show()


    normal_data.drop('sum_flags', axis=1, inplace=True)
    abnormal_data.drop('sum_flags', axis=1, inplace=True)
    grey_data.drop('sum_flags', axis=1, inplace=True)
    grey_data_train.drop('sum_flags', axis=1, inplace=True)

    normal_data_train, normal_data_test = train_test_split(normal_data, test_size=test_size, random_state=1234)
    train_data = pd.concat([normal_data_train,grey_data_train])

    abnormal_data_index = abnormal_data.index
    normal_data_train_index = normal_data_train.index
    normal_data_test_index = normal_data_test.index
    grey_data_index = grey_data.index
    grey_data_train_index = grey_data_train.index
    train_data_index = train_data.index
    

    scaler = StandardScaler()
    scaler.fit(train_data)

    train_data_scl = scaler.transform(train_data)
    normal_data_test_scl = scaler.transform(normal_data_test)
    abnormal_data_scl = scaler.transform(abnormal_data)


    print(f'Normal Train Dataset: {normal_data_train.shape}')
    print(f'Grey Train Dataset: {grey_data_train.shape}')
    print(f'Train Dataset Normal + Grey: {train_data_scl.shape}')
    print(f'Normal Test Dataset: {normal_data_test_scl.shape}')
    print(f'Abnormal Test Dataset: {abnormal_data_scl.shape}')
    print(f'Grey Dataset: {grey_data.shape}')



    train_dataset = MLP_AE_dataset(dataframe=train_data_scl) 
    test_normal_dataset = MLP_AE_dataset(dataframe=normal_data_test_scl) 
    test_abnormal_dataset = MLP_AE_dataset(dataframe=abnormal_data_scl) 

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=shuffle)
    test_dataloader  = DataLoader(test_normal_dataset, batch_size=batch_size_test)
    abnormal_dataloader  = DataLoader(test_abnormal_dataset, batch_size=batch_size_test)
    
    
    
    return {'abnormal_data_index':abnormal_data_index,
            'normal_data_train_index':normal_data_train_index,
            'normal_data_test_index':normal_data_test_index,
            'grey_data_index':grey_data_index,
            'grey_data_train_index':grey_data_train_index,
            'train_data_index':train_data_index,
            'fit_scaler_fun':scaler,
            'train_dataloader':train_dataloader,
            'test_dataloader':test_dataloader,
            'abnormal_dataloader':abnormal_dataloader}
    
# def data_provider(dataframe, test_size, normal_sum_flags_threshold, batch_size_train, batch_size_test, shuffle):
#     '''
    
#     {'abnormal_data_index':abnormal_data_index,
#     'normal_data_train_index':normal_data_train_index,
#     'normal_data_test_index':normal_data_test_index,
#     'fit_scaler_fun':scaler,
#     'train_dataloader':train_dataloader,
#     'test_dataloader':test_dataloader,
#     'test_dataloader_abnormal':test_dataloader_abnormal} = data_provider(dataframe, test_size, normal_sum_flags_threshold, batch_size_train, batch_size_test, shuffle)
    
#     '''
#     print(f'Shape of dataframe {dataframe.shape}')
#     # Drop rows with Null cell. 
#     dataframe = dataframe.drop(dataframe[dataframe.isnull().any(axis=1)].index)
#     print(f'Shape of dataframe after drop null data {dataframe.shape}')

#     normal_data   = dataframe[dataframe.sum_flags <  normal_sum_flags_threshold]
#     abnormal_data = dataframe[dataframe.sum_flags >= normal_sum_flags_threshold]
    
#     normal_data[  ['sum_flags']].hist()
#     abnormal_data[['sum_flags']].hist()

#     normal_data.drop('sum_flags', axis=1, inplace=True)
#     abnormal_data.drop('sum_flags', axis=1, inplace=True)

#     normal_data_train, normal_data_test = train_test_split(normal_data, test_size=test_size)

#     abnormal_data_index = abnormal_data.index
#     normal_data_train_index = normal_data_train.index
#     normal_data_test_index = normal_data_test.index

#     scaler = StandardScaler()
#     scaler.fit(normal_data_train)
#     normal_data_train_scl = scaler.transform(normal_data_train)
#     normal_data_test_scl = scaler.transform(normal_data_test)
#     abnormal_data_test_scl = scaler.transform(abnormal_data)


#     print(f'Normal Train Dataset: {normal_data_train_scl.shape}')
#     print(f'Normal Test Dataset: {normal_data_test_scl.shape}')
#     print(f'Abnormal Train Dataset: {abnormal_data_test_scl.shape}')


#     train_normal_data = my_dataset(dataframe=normal_data_train_scl) 
#     test_normal_data = my_dataset(dataframe=normal_data_test_scl) 
#     test_abnormal_data = my_dataset(dataframe=abnormal_data_test_scl) 

#     train_dataloader = DataLoader(train_normal_data, batch_size=batch_size_train, shuffle=shuffle)
#     test_dataloader  = DataLoader(test_normal_data, batch_size=batch_size_test)
#     test_dataloader_abnormal  = DataLoader(test_abnormal_data, batch_size=batch_size_test)
#     return {'abnormal_data_index':abnormal_data_index,
#             'normal_data_train_index':normal_data_train_index,
#             'normal_data_test_index':normal_data_test_index,
#             'fit_scaler_fun':scaler,
#             'train_dataloader':train_dataloader,
#             'test_dataloader':test_dataloader,
#             'test_dataloader_abnormal':test_dataloader_abnormal}


################################
#           MLP Model          #
################################

# class MLP_autoencoder1(nn.Module):
#     def __init__(self, input_size, dropout):
#         super(MLP_autoencoder1, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Linear(input_size, 10),
#             nn.ReLU(),
#             nn.Linear(10, 5),
#             nn.ReLU(),
#             nn.Linear(5, 3),
#             nn.ReLU())
        
        
#         self.decoder = nn.Sequential(
#             nn.Linear(3, 5),
#             nn.ReLU(),
#             nn.Linear(5, 10),
#             nn.ReLU(),
#             nn.Linear(10, input_size),
#             nn.ReLU())
        
#     def forward(self, x):
#         x = self.encoder(x)
# #         print(f'x.shape {x.shape}')
# #         print(f'x.size {x.size()}')
#         x = self.decoder(x)
#         return x
    
    
class MLP_autoencoder2(nn.Module):
    def __init__(self, input_size, dropout):
        super(MLP_autoencoder2, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 150),
            nn.ReLU(),
            nn.Linear(150, 50),
            nn.ReLU(),
            nn.Linear(50, 10),
            nn.ReLU())
        
        
        self.decoder = nn.Sequential(
            nn.Linear(10, 50),
            nn.ReLU(),
            nn.Linear(50, 150),
            nn.ReLU(),
            nn.Linear(150, input_size),
            nn.ReLU())
        
    def forward(self, x):
        x = self.encoder(x)
#         print(f'x.shape {x.shape}')
#         print(f'x.size {x.size()}')
        x = self.decoder(x)
        return x
    
    
    
# class MLP_autoencoder3(nn.Module):
#     def __init__(self, input_size, dropout):
#         super(MLP_autoencoder3, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Linear(input_size, 150),
#             nn.ReLU(),
#             nn.Linear(150, 100),
#             nn.ReLU(),
#             nn.Linear(100, 50),
#             nn.ReLU(),
#             nn.Linear(50, 10),
#             nn.ReLU())
        
        
#         self.decoder = nn.Sequential(
#             nn.Linear(10, 50),
#             nn.ReLU(),
#             nn.Linear(50, 100),
#             nn.ReLU(),
#             nn.Linear(100, 150),
#             nn.ReLU(),
#             nn.Linear(150, input_size),
#             nn.ReLU())
        
#     def forward(self, x):
#         x = self.encoder(x)
# #         print(f'x.shape {x.shape}')
# #         print(f'x.size {x.size()}')
#         x = self.decoder(x)
#         return x
    
    
    
# def train_loop(dataloader, model, criterion, optimizer,device):
#     loss = 0
#     for batch, (X,y) in enumerate(dataloader):
#         X = X.to(device).float()
#         y = y.to(device).float()
# #         display('X',X.size())
# #         display('y',y.size())
        
#         pred = model(X)
# #         display('pred',pred.size())
#         loss_batch = criterion(pred, y)
#         optimizer.zero_grad()
#         loss_batch.backward()
#         optimizer.step()
#         loss += loss_batch.item()
#     loss /= len(dataloader.dataset)
    
#     return loss
    

# def test_loop(dataloader, model, criterion,device):
#     loss = 0
#     with torch.no_grad():
#         for batch, (X,y) in enumerate(dataloader):
#             X = X.to(device).float()
#             y = y.to(device).float()
#             pred = model(X)
#             loss += criterion(pred, y).item()
#         loss /= len(dataloader.dataset)
#     return loss 




def train_loop(dataloader, model, criterion, optimizer,device):
    loss = 0
    for batch, (X,y) in enumerate(dataloader):
        X = X.to(device).float()
        y = y.to(device).float()
#         display('X',X.size())
#         display('y',y.size())
        
        pred = model(X)
#         display('pred',pred.size())
        loss_batch = criterion(pred, y)
        optimizer.zero_grad()
        loss_batch.backward()
        optimizer.step()
        loss += loss_batch.item()*len(X)
    loss /= len(dataloader.dataset)
    
    return loss
    

def test_loop(dataloader, model, criterion,device):
    loss = 0
    with torch.no_grad():
        for batch, (X,y) in enumerate(dataloader):
            X = X.to(device).float()
            y = y.to(device).float()
            pred = model(X)
            loss += criterion(pred, y).item()*len(X)
        loss /= len(dataloader.dataset)
    return loss 



#########################################################
#                          LSTM                         #
#########################################################

################################
#        LSTM DataLoader       #
################################

def cancel_index_with_null(dataframe, Time_Window, total_null_threshold=None, row_null_threshold=None, verbose=False):
    verboseprint = print if verbose else lambda *a, **k: None
    null_index = []

    
    index_list_null_removed = dataframe.index
    
    for idx in dataframe.index:
        drop_idx_flag = True
        df_TW = dataframe.loc[idx-Time_Window+datetime.timedelta(seconds=1):idx, :]
        
        if  total_null_threshold: 
            if df_TW.isnull().sum(axis=1).sum() >= total_null_threshold:
                
                if drop_idx_flag:
                    index_list_null_removed = index_list_null_removed.drop(idx)
                    drop_idx_flag = False
                
                verboseprint('Drop due to null total sum ====>', idx,
                      'Null total sum:',df_TW.isnull().sum(axis=1).sum(),
                      'len of datafram ',len(dataframe.index),
                      'len of index_list_null_removed ',len(index_list_null_removed),
                      'df_TW shape',df_TW.shape
                     )
    
    
        if row_null_threshold:
            for i in df_TW.index:
                if df_TW.loc[i, :].isnull().sum() >= row_null_threshold:
                    
                    if drop_idx_flag:
                        index_list_null_removed = index_list_null_removed.drop(idx)
                        drop_idx_flag = False
                
                
                    verboseprint(5*'=+','>',
                          'Drop due to null row sum ====>', idx,
                          'Null row sum:',df_TW.loc[i, :].isnull().sum(),
                          'len of datafram ',len(dataframe.index),
                          'len of index_list_null_removed ',len(index_list_null_removed),
                          'df_TW shape',df_TW.shape)
        if (not drop_idx_flag):
            null_index.append(idx)
            
            
            
    assert (not set(null_index).intersection(set(index_list_null_removed))), 'There are same index in both list !!!'
    
    return index_list_null_removed, pd.to_datetime(null_index) 

class LSTM_AE_dataset(Dataset):
    
    def __init__(self, dataframe, Time_Window, Index_lst):
        self.dataframe = dataframe
        self.Time_Window = Time_Window
        self.Index_lst = Index_lst
        
    def _one_timeseries(self, idx):
        
        '''
        return pandas DataFrame
        rows are futuers columns are timestamp
        _one_timeseries(dataframe=ambient_dt, Time_Window=datetime.timedelta(hours=6), index=pd.to_datetime('2021-06-22 23:30:00'))
        output: rows are futuers columns are timestamp
        return transposed dataframe start from end_index - time_window and end end_index  
        '''
#         print(self.Index_lst[idx])
        end_index = self.Index_lst[idx]
        

#         print(f'start:{end_index-self.Time_Window+datetime.timedelta(seconds=1)} and end {end_index}')
#             display(self.dataframe.loc[end_index-self.Time_Window+datetime.timedelta(seconds=1):end_index, :].T)
        return self.dataframe.loc[end_index-self.Time_Window+datetime.timedelta(seconds=1):end_index, :].T
            
        
    def _convert_to_numpy(self, df):
        '''
        return the numpy fromat of data and dictionary that conatin the index and coulmns of the dataframe
        '''
#         dic_cols_rows = {'index':'', #self.dataframe.index.values
#                          'columns':}
#         display(df, df.index)
        return df.to_numpy(), list(df.index.values), list(df.columns.strftime('%Y-%m-%d %H:%M:%S'))
     
    
    def __len__(self):
        return len(self.Index_lst)
    
    
    def __getitem__(self, idx):
        '''
        idx is int between the zero and __len__(self) so I convert it to datetime in _one_timeseries() 
        end_index = self.Index_lst[idx] 
        ''' 
        return self._convert_to_numpy(self._one_timeseries(idx))
    
    
    

# def LSTM_data_provider(dataframe, test_size, normal_sum_flags_threshold, batch_size_train, batch_size_test, shuffle,Time_Window):
#     '''
    
#     {'abnormal_data_index':abnormal_data_index,
#     'normal_data_train_index':normal_data_train_index,
#     'normal_data_test_index':normal_data_test_index,
#     'fit_scaler_fun':scaler,
#     'train_dataloader':train_dataloader,
#     'test_dataloader':test_dataloader,
#     'test_dataloader_abnormal':test_dataloader_abnormal} = data_provider(dataframe, test_size, normal_sum_flags_threshold, batch_size_train, batch_size_test, shuffle)
    
#     '''
# #     print(f'Shape of dataframe {dataframe.shape}')
#     # Drop rows with Null cell. 
# #     dataframe = dataframe.drop(dataframe[dataframe.isnull().any(axis=1)].index)
# #     print(f'Shape of dataframe after drop null data {dataframe.shape}')


#     def _normal_abnormal_index(dataframe, normal_sum_flags_threshold):
#         normal_data   = dataframe[dataframe.sum_flags <  normal_sum_flags_threshold]
#         abnormal_data = dataframe[dataframe.sum_flags >= normal_sum_flags_threshold]

#         normal_data[  ['sum_flags']].hist()
#         abnormal_data[['sum_flags']].hist()
#         print(f'Size of normal dataset:{len(normal_data.index)}  Size of abnormal dataset:{len(abnormal_data.index)}')
#         return normal_data.index, abnormal_data.index

#     normal_data_index, abnormal_data_index = _normal_abnormal_index(dataframe=dataframe, normal_sum_flags_threshold=normal_sum_flags_threshold)
#     # Filter out the firts 6 hours index form begining of them dataframe. 
#     normal_data_index = normal_data_index[normal_data_index>dataframe.index[37]]
#     abnormal_data_index = abnormal_data_index[abnormal_data_index>dataframe.index[37]]
#     print(f'Size of normal dataset: {len(normal_data_index)},  Size of abnormal dataset: {len(abnormal_data_index)}')
    
#     normal_data_train_index, normal_data_test_index = train_test_split(normal_data_index, test_size=test_size, random_state=1234)

    

#     scaler = StandardScaler()
#     scaler.fit(dataframe.loc[normal_data_train_index].drop('sum_flags', axis=1))
#     dataframe_scl = pd.DataFrame(scaler.transform(dataframe.drop('sum_flags', axis=1)), columns=dataframe.drop('sum_flags', axis=1).columns)
#     dataframe_scl.index = dataframe.index
    

#     train_normal_dataset = dataset_LSTM(dataframe=dataframe_scl, Time_Window=Time_Window, Index_lst=normal_data_train_index)
#     test_normal_dataset = dataset_LSTM(dataframe=dataframe_scl, Time_Window=Time_Window, Index_lst=normal_data_test_index)
#     abnormal_dataset = dataset_LSTM(dataframe=dataframe_scl, Time_Window=Time_Window, Index_lst=abnormal_data_index)

    
#     train_dataloader = DataLoader(train_normal_dataset, batch_size=batch_size_train, shuffle=shuffle)
#     test_dataloader = DataLoader(test_normal_dataset, batch_size=batch_size_test)
#     abnormal_dataloader = DataLoader(abnormal_dataset, batch_size=batch_size_test)
    
    
#     return {'abnormal_data_index':abnormal_data_index,
#             'normal_data_train_index':normal_data_train_index,
#             'normal_data_test_index':normal_data_test_index,
#             'fit_scaler_fun':scaler,
#             'train_dataloader':train_dataloader,
#             'test_dataloader':test_dataloader,
#             'abnormal_dataloader':abnormal_dataloader}



def LSTM_AE_data_provider(dataframe, test_size, grey_data_range, grey_data_train_threshold, batch_size_train, batch_size_test, shuffle, Time_Window=datetime.timedelta(minutes=20), total_null_threshold=12, row_null_threshold=6,verbose=False):
    
    '''
        return {'abnormal_data_index':abnormal_data_index,
                'normal_data_train_index':normal_data_train_index,
                'normal_data_test_index':normal_data_test_index,
                'grey_data_index':grey_data_index,
                'grey_data_train_index':grey_data_train_index,
                'train_data_index':train_data_index,
                'fit_scaler_fun':scaler,
                'train_dataloader':train_dataloader,
                'test_dataloader':test_dataloader,
                'abnormal_dataloader':abnormal_dataloader}
    '''
            
#     grey_data_range = [1,25]

    verboseprint   = print   if verbose else lambda *a, **k: None
    verbosedisplay = display if verbose else lambda *a, **k: None
    

    index_list_null_removed, null_index = cancel_index_with_null(dataframe.drop('sum_flags', axis=1), Time_Window=Time_Window, total_null_threshold=total_null_threshold, row_null_threshold=row_null_threshold, verbose=verbose)
    
    
    normal_data   = dataframe.query("sum_flags < @grey_data_range[0]")
    abnormal_data = dataframe.query("sum_flags > @grey_data_range[1]")
    grey_data     = dataframe.query("@grey_data_range[0] <= sum_flags <= @grey_data_range[1]")
    grey_data_train=dataframe.query("@grey_data_range[0] <= sum_flags <= @grey_data_train_threshold")

    #   Plots
    verboseprint('Normal Dataset Sum of Flags')
    normal_data[  ['sum_flags']].hist()
    plt.show()
    verboseprint('Abnormal Dataset Sum of Flags')
    abnormal_data[['sum_flags']].hist()
    plt.show()
    verboseprint('Grey Dataset Sum of Flags')
    grey_data[['sum_flags']].hist()
    plt.show()
    verboseprint('Grey Train Dataset Sum of Flags')
    grey_data_train[['sum_flags']].hist()
    plt.show()

    normal_data.drop('sum_flags', axis=1, inplace=True)
    abnormal_data.drop('sum_flags', axis=1, inplace=True)
    grey_data.drop('sum_flags', axis=1, inplace=True)
    grey_data_train.drop('sum_flags', axis=1, inplace=True)


    normal_data_index     = normal_data.index   
    abnormal_data_index   = abnormal_data.index 
    grey_data_index       = grey_data.index     
    grey_data_train_index = grey_data_train.index
    
    
    normal_data_index = pd.to_datetime([x for x in normal_data_index if x not in null_index])
    grey_data_train_index = pd.to_datetime([x for x in grey_data_train_index if x not in null_index])
    grey_data_index = pd.to_datetime([x for x in grey_data_index if x not in null_index])
    abnormal_data_index = pd.to_datetime([x for x in abnormal_data_index if x not in null_index])
    
#     print(normal_data_index)

    # Filter out the firts 6 hours index form begining of them dataframe. 
    normal_data_index     = normal_data_index[normal_data_index>dataframe.index[37]]
    abnormal_data_index   = abnormal_data_index[abnormal_data_index>dataframe.index[37]]
    grey_data_index       = grey_data_index[grey_data_index>dataframe.index[37]]
    grey_data_train_index = grey_data_train_index[grey_data_train_index >dataframe.index[37]]


    
    
    
    normal_data_train_index, normal_data_test_index = train_test_split(normal_data_index, test_size=test_size, random_state=1234)
    train_data_index = grey_data_train_index.append(normal_data_train_index)



    print(f'Normal Train Index: {len(normal_data_train_index)}')
    print(f'Grey Train Index: {len(grey_data_train_index)}')
    print(f'Train Normal + Grey Index: {len(train_data_index)}')
    print(f'Normal Test Index: {len(normal_data_test_index)}')
    print(f'Abnormal Test Index: {len(abnormal_data_index)}')
    print(f'Grey Index: {len(grey_data_index)}')

    
    
    print(f'Sum of null {dataframe.isnull().sum(axis=1).sum()}')
    dataframe = dataframe.interpolate()
    print(f'Sum of null after interpolation: {dataframe.isnull().sum(axis=1).sum()}')
    

    scaler = StandardScaler()
    scaler.fit(dataframe.loc[train_data_index,:].drop('sum_flags', axis=1))
    dataframe_scl = pd.DataFrame(scaler.transform(dataframe.drop('sum_flags', axis=1)), columns=dataframe.drop('sum_flags', axis=1).columns)
    dataframe_scl.index = dataframe.index
                           
    train_normal_dataset = LSTM_AE_dataset(dataframe=dataframe_scl, Time_Window=Time_Window, Index_lst=train_data_index)
    test_normal_dataset = LSTM_AE_dataset(dataframe=dataframe_scl, Time_Window=Time_Window, Index_lst=normal_data_test_index)
    abnormal_dataset = LSTM_AE_dataset(dataframe=dataframe_scl, Time_Window=Time_Window, Index_lst=abnormal_data_index)


    train_dataloader = DataLoader(train_normal_dataset, batch_size=batch_size_train, shuffle=shuffle)
    test_dataloader = DataLoader(test_normal_dataset, batch_size=batch_size_test)
    abnormal_dataloader = DataLoader(abnormal_dataset, batch_size=batch_size_test)
    
    return {'abnormal_data_index':abnormal_data_index,
            'normal_data_train_index':normal_data_train_index,
            'normal_data_test_index':normal_data_test_index,
            'grey_data_index':grey_data_index,
            'grey_data_train_index':grey_data_train_index,
            'train_data_index':train_data_index,
            'fit_scaler_fun':scaler,
            'train_dataloader':train_dataloader,
            'test_dataloader':test_dataloader,
            'abnormal_dataloader':abnormal_dataloader,
            'null_index':null_index}





################################
#           LSTM Model         #
################################


class Encoder(nn.Module):

  def __init__(self, seq_len, n_features, embedding_dim=64):
    super(Encoder, self).__init__()

    self.seq_len, self.n_features = seq_len, n_features
    self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim

    self.rnn1 = nn.LSTM(input_size=n_features,hidden_size=self.hidden_dim,num_layers=1,batch_first=True)
    self.rnn2 = nn.LSTM(input_size=self.hidden_dim,hidden_size=embedding_dim,num_layers=1,batch_first=True)

  def forward(self, x):
#     print('Encoder')
#     print(f'Size of input: {x.size()}')
    x = x.reshape((1, self.seq_len, self.n_features))
#     print(f'Reshape(1, self.seq_len, self.n_features): {x.size()}')
    
    x, (_, _) = self.rnn1(x)
#     print(f'Out of lstm-1: {x.size()}')
    x, (hidden_n, c_n) = self.rnn2(x)
#     print(f'Out of lstm-2: x:{x.size()} h:{hidden_n.size()} c:{c_n.size()}')
    out = hidden_n.squeeze(0)
#     print(f'Out of encoder: {out.size()}')
#     print(25*'+-+-')
    return out 


class Decoder(nn.Module):

  def __init__(self, seq_len, input_dim=64, n_features=5):
    super(Decoder, self).__init__()

    self.seq_len, self.input_dim = seq_len, input_dim
    self.hidden_dim, self.n_features = 2 * input_dim, n_features

    self.rnn1 = nn.LSTM(input_size=input_dim,hidden_size=input_dim,num_layers=1,batch_first=True)
    self.rnn2 = nn.LSTM(input_size=input_dim,hidden_size=self.hidden_dim,num_layers=1,batch_first=True)
    self.output_layer = nn.Linear(self.hidden_dim, n_features)

  def forward(self, x):
#     print('Decoder')
#     print(f'Input of Decoder: {x.size()}')
    x = x.repeat(self.seq_len, 1)
#     print(f'Repeat(L,H0): {x.size()}, {x}')
    x = x.reshape((1, self.seq_len, self.input_dim))
#     print(f'Reshape(H0,L,inpu_dim): {x.size()}{x}')
    

    x, (hidden_n, cell_n) = self.rnn1(x)
#     print(f'Out of lstm-1: {x.size()}')
    x, (hidden_n, cell_n) = self.rnn2(x)
#     print(f'Out of lstm-2: x:{x.size()}')
#     x = torch.transpose(x,0,2)
#     x = torch.transpose(x,0,1)
#     x = x.reshape((self.seq_len, self.hidden_dim))
#     print(f'Reshape(L,H1): {x.size()}')
    out = self.output_layer(x)
#     print(f'Out of decoder: {out.size()}')
    return out



class RecurrentAutoencoder(nn.Module):

  def __init__(self, seq_len, n_features, embedding_dim=64):
    super(RecurrentAutoencoder, self).__init__()

    self.encoder = Encoder(seq_len, n_features, embedding_dim)
    self.decoder = Decoder(seq_len, embedding_dim, n_features)

  def forward(self, x):
    x = self.encoder(x)
    x = self.decoder(x)

    return x


def lstm_train_loop(dataloader, model, criterion, optimizer,device):
    loss = 0
    for batch, X in enumerate(dataloader):
        X = torch.transpose(X[0],1,2)
        X = X.to(device).float()
#         display('X',X.size())
        pred = model(X)
#         display('pred',pred.size())
        loss_batch = criterion(pred, X)
        optimizer.zero_grad()
        loss_batch.backward()
        optimizer.step()
        loss += loss_batch.item()
    loss /= len(dataloader.dataset)
    
    return loss
    

def lstm_test_loop(dataloader, model, criterion,device):
    loss = 0
    with torch.no_grad():
        for batch, X in enumerate(dataloader):
            X = torch.transpose(X[0],1,2)
            X = X.to(device).float()
            pred = model(X)
            loss += criterion(pred, X).item()
        loss /= len(dataloader.dataset)
    return loss



























































# import os, sys, imp
# sys.path.append('/home/seyedkazemi/codes/mskhelper/')
# sys.path.append('/home/seyedkazemi/codes/Marconi100/DNN/')
# import pandas as pd, numpy as np
# import datetime, pytz
# import data_provider, mohsenutils, m100_preprocessing_helper, flag_helper, autoencoder_helper
# import imp 
# from  matplotlib import pyplot as plt
# import torch 
# import torch.nn as nn
# import torch.functional as f
# from torch.utils.data import DataLoader, Dataset
# from torchinfo import summary
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from contextlib import redirect_stdout
# import json
# import datetime, pytz
# import random
# import function_set as fn
# from contextlib import redirect_stdout
# from pickle import dump
# imp.reload(data_provider)
# imp.reload(mohsenutils)
# imp.reload(m100_preprocessing_helper)
# imp.reload(flag_helper)
# imp.reload(autoencoder_helper)


# def LSTM_AE_Complete(rack = '205',
#                      test_size = 0.25,
#                      grey_data_range = [16,35],
#                      grey_data_train_threshold = 15,
#                      batch_size_train = 1,
#                      batch_size_test = 1,
#                      shuffle = True,
#                      epochs = 100,
#                      learning_rate = 1e-3,
#                      dropout = 0,
#                      cuda='3',
#                      Time_Window = datetime.timedelta(minutes=20)):


#     SEED = 1234
#     random.seed(SEED)
#     np.random.seed(SEED)
#     torch.manual_seed(SEED)
#     torch.cuda.manual_seed(SEED)
#     torch.backends.cudnn.deterministic = True

#     now = datetime.datetime.now(pytz.timezone('Europe/Rome'))
#     datename = str(now.year)+str(now.month)+str(now.day)+str(now.hour)+str(now.minute)+str(now.second)
    
#     device = torch.device('cuda:'+str(cuda) if torch.cuda.is_available() else 'cpu')
# #     device = torch.device('cpu')
    
#     sequence_length = int(Time_Window/datetime.timedelta(minutes=10))
    




#     print(10*'~+~')
#     print(f'The sequence length is :{sequence_length}')
#     print(10*'~+~')



#     dataframe = autoencoder_helper.flags_data_new(rack=rack, 
#                                           ambient_dt=True,
#                                           pcie_dt=True,
#                                           fsnd_dt=True,
#                                           psnd_dt=True,
#                                           gpu0_dt=True,
#                                           gpu1_dt=True,
#                                           gpu3_dt=True,
#                                           gpu4_dt=True,
#                                           cpu0_dt=True,
#                                           cpu1_dt=True,
#                                           Tot_ict_dt=True,
#                                           Tot_qpompe_dt=True, 
#                                           RDHX_dt=True,
#                                           Tot_chiller_dt=True,
#                                           Supply_Air_Temperature_dt=True,
#                                           Return_Air_Temperature_dt=True, 
#                                           Free_Cooling_Valve_Open_Position_dt=True,
#                                           Tot_cdz_dt=True, 
#                                           FScrac_dt=True,
#                                           Free_Cooling_Status_dt=True,
#                                           Compressor_Utilization_dt=True,
#                                           temp_dt=True)



#     dt_provider = autoencoder_helper.LSTM_AE_data_provider(dataframe=dataframe, 
#                                                     test_size=test_size,
#                                                     grey_data_range=grey_data_range,  
#                                                     grey_data_train_threshold=grey_data_train_threshold, 
#                                                     batch_size_train=batch_size_train, 
#                                                     batch_size_test=batch_size_test, 
#                                                     shuffle=shuffle,
#                                                     Time_Window=Time_Window)



#     model = autoencoder_helper.RecurrentAutoencoder(seq_len=sequence_length, n_features=dataframe.shape[1]-1, embedding_dim=64).float()
#     model_summary = summary(model.float(),(1,sequence_length,dataframe.shape[1]-1))
#     display(model)
#     display(model_summary) 
#     model = model.to(device)

#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#     criterion = torch.nn.L1Loss(reduction='sum').to(device)



#     restults_complete={'Test_Size':test_size,
#                        'grey_data_range':str(grey_data_range),
#                        'grey_data_train_threshold':grey_data_train_threshold,
#                        'Batch_Size_Train':batch_size_train,
#                        'Batch_Size_Test':batch_size_test,
#                        'Shuffle':shuffle,
#                        'Rack':rack,
#                        'Number_of_Epochs':epochs,
#                        'Learning_Rate':learning_rate, 
#                        'Device':str(device), 
#                        'Criterion':'torch.nn.L1Loss(reduction=sum)',
#                        'Optimizer':'torch.optim.Adam(model.parameters(), lr=learning_rate)',
#                        'abnormal_data_index':list(dt_provider['abnormal_data_index'].astype(str).values),
#                        'normal_data_train_index':list(dt_provider['normal_data_train_index'].astype(str).values),
#                        'normal_data_test_index':list(dt_provider['normal_data_test_index'].astype(str).values),
#                        'grey_data_index':list(dt_provider['grey_data_index'].astype(str).values),
#                        'grey_data_train_index':list(dt_provider['grey_data_train_index'].astype(str).values),
#                        'train_data_index':list(dt_provider['train_data_index'].astype(str).values),
#                        'Number_featuers':dataframe.shape[1]-1,
#                        'columns':list(dataframe.columns.values),
#                        'Time_Window':str(Time_Window).replace(':','_'),
#                        'Epochs_Results':[],
#                        'null_index':list(dt_provider['null_index'].astype(str).values)}




#     dir_path = '/home/seyedkazemi/dataset/Marconi100-Cooked/AE/LSTM-AE-4/LSTM-AE_Results_SqL'+str(sequence_length)+'-ftur'+str(dataframe.shape[1]-1)+'grey_data_range_'+str(grey_data_range[0]).zfill(2)+'_'+str(grey_data_range[1]).zfill(2)+'-f'+str(grey_data_train_threshold).zfill(2)+'/'
#     if not os.path.isdir(dir_path):
#         os.makedirs(dir_path)


#     # save the scaler
#     dump(dt_provider['fit_scaler_fun'], open(dir_path+'scaler.pkl', 'wb'))
#     display(dt_provider)


#     for epoch in range(1, epochs+1):
#         model.train()
#         train_loss = autoencoder_helper.lstm_train_loop(dt_provider['train_dataloader'], model, criterion, optimizer,device=device)
#         model.eval()
#         test_loss = autoencoder_helper.lstm_test_loop(dt_provider['test_dataloader'], model, criterion,device=device)
#         test_loss_abnormal = autoencoder_helper.lstm_test_loop(dt_provider['abnormal_dataloader'], model, criterion,device=device)    
#         restults_complete['Epochs_Results'].append({'epoch':epoch,'TrainLoss':train_loss,'TestLoss':test_loss,'TestAbnormalLoss':test_loss_abnormal})

#         if epoch%10==0:
#             print(f"{epoch} Train loss: {train_loss:>7f} and Test loss: {test_loss:>7f} and Abnormal Test loss: {test_loss_abnormal:>7f}")


#         if epoch%10==0:
#             model_name = mohsenutils.file_name_creator(path=dir_path, file_name='model_epoch_'+str(epoch)+'.pt', log=False)
#             torch.save(model, dir_path+model_name)    

#     output_file_name = mohsenutils.file_name_creator(path=dir_path, file_name='output.txt', log=False)

#     with open(dir_path+output_file_name, 'w') as outfile:  
#         outfile.write(str(summary(model.float(),(1,sequence_length,dataframe.shape[1]-1))))
#         outfile.write('#+#')
#         json.dump(restults_complete, outfile)
#         outfile.close()



#     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#     # INFERENCE RESULTS    
#     dataframe_scl = pd.DataFrame(dt_provider['fit_scaler_fun'].transform(dataframe.drop('sum_flags', axis=1)), columns=dataframe.drop('sum_flags', axis=1).columns)
#     dataframe_scl.index = dataframe.index
#     all_dataset = autoencoder_helper.LSTM_AE_dataset(dataframe=dataframe_scl, Time_Window=Time_Window, Index_lst=dataframe_scl.iloc[37:,:].index)

#     inference_results = pd.DataFrame(columns=['Error'])

#     for dtl_idx, X in enumerate(DataLoader(all_dataset)):
#         model.eval()
#         dt, cols, idx = X
#         x = torch.transpose(dt,1,2).to(device).float()
#         if dtl_idx == 0 :
#             print(f'Input X size is :{x.size()}\nInput X[0]is:\n\t{x}')
#         pred = model(x)
#         loss = criterion(pred, x)
#         inference_results.loc[idx[-1][0],'Error']=loss.item()


#     inference_results.index.name = 'timestamp'
#     inference_results.index = pd.to_datetime(inference_results.index)
#     inference_results['Label'] =''
#     inference_results.loc[dt_provider['normal_data_train_index'],'Label'] = 'train'
#     inference_results.loc[dt_provider['normal_data_test_index'],'Label'] = 'test'
#     inference_results.loc[dt_provider['abnormal_data_index'],'Label'] = 'abnormal'

#     inference_results.loc[dt_provider['grey_data_index'],'Label'] = 'grey'
#     inference_results.loc[dt_provider['grey_data_train_index'],'Label'] = 'grey_train'


#     inference_results = mohsenutils.mrg(None,'inner',dataframe[['sum_flags']], inference_results)
#     inference_results.to_csv(dir_path+'Inference_Results.csv') 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
import os, sys, imp
sys.path.append('/home/seyedkazemi/codes/mskhelper/')
sys.path.append('/home/seyedkazemi/codes/Marconi100/DNN/')
import pandas as pd, numpy as np
import datetime, pytz
import data_provider, mohsenutils, m100_preprocessing_helper, flag_helper, autoencoder_helper
import imp 
from  matplotlib import pyplot as plt
import torch 
import torch.nn as nn
import torch.functional as f
from torch.utils.data import DataLoader, Dataset
from torchinfo import summary
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from contextlib import redirect_stdout
import json
import datetime, pytz
import random
import function_set as fn
from contextlib import redirect_stdout
from pickle import dump
imp.reload(data_provider)
imp.reload(mohsenutils)
imp.reload(m100_preprocessing_helper)
imp.reload(flag_helper)
imp.reload(autoencoder_helper)


def LSTM_AE_Complete_Monthly(dir_name,
                             rack = '205',
                             test_size = 0.000000001,
                             batch_size_train = 1,
                             batch_size_test = 1,
                             shuffle = True,
                             epochs = 100,
                             learning_rate = 1e-3,
                             dropout = 0,
                             cuda='3',
                             Time_Window = datetime.timedelta(minutes=20),
                             start_train='2021-06-15 00:00:00',
                             stop_train ='2021-07-15 00:00:00',
                             grey_data_range = [11,35],
                             grey_data_train_threshold = 30):


    SEED = 1234
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    now = datetime.datetime.now(pytz.timezone('Europe/Rome'))
    datename = str(now.year)+str(now.month)+str(now.day)+str(now.hour)+str(now.minute)+str(now.second)

    device = torch.device('cuda:'+str(cuda) if torch.cuda.is_available() else 'cpu')
    
    sequence_length = int(Time_Window/datetime.timedelta(minutes=10))



    print(10*'~+~')
    print(f'The sequence length is :{sequence_length}')
    print(10*'~+~')



    dataframe = flags_data_new(rack=rack, 
                                              ambient_dt=True,
                                              pcie_dt=True,
                                              fsnd_dt=True,
                                              psnd_dt=True,
                                              gpu0_dt=True,
                                              gpu1_dt=True,
                                              gpu3_dt=True,
                                              gpu4_dt=True,
                                              cpu0_dt=True,
                                              cpu1_dt=True,
                                              Tot_ict_dt=True,
                                              Tot_qpompe_dt=True, 
                                              RDHX_dt=True,
                                              Tot_chiller_dt=True,
                                              Supply_Air_Temperature_dt=True,
                                              Return_Air_Temperature_dt=True, 
                                              Free_Cooling_Valve_Open_Position_dt=True,
                                              Tot_cdz_dt=True, 
                                              FScrac_dt=True,
                                              Free_Cooling_Status_dt=True,
                                              Compressor_Utilization_dt=True,
                                              temp_dt=True)



    dataframe = dataframe.query('@start_train<=@dataframe.index<@stop_train')
    display(dataframe)

    dt_provider = LSTM_AE_data_provider(dataframe=dataframe, 
                                                        test_size=test_size,
                                                        grey_data_range=grey_data_range,  
                                                        grey_data_train_threshold=grey_data_train_threshold, 
                                                        batch_size_train=batch_size_train, 
                                                        batch_size_test=batch_size_test, 
                                                        shuffle=shuffle,
                                                        Time_Window=Time_Window)


    model = RecurrentAutoencoder(seq_len=sequence_length, n_features=dataframe.shape[1]-1, embedding_dim=64).float()
    model_summary = summary(model.float(),(1, sequence_length, dataframe.shape[1]-1))
    display(model)
    display(model_summary) 
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.L1Loss(reduction='sum').to(device)



    restults_complete={'Test_Size':test_size,
                       'grey_data_range':str(grey_data_range),
                       'grey_data_train_threshold':grey_data_train_threshold,
                       'Batch_Size_Train':batch_size_train,
                       'Batch_Size_Test':batch_size_test,
                       'Shuffle':shuffle,
                       'Rack':rack,
                       'Number_of_Epochs':epochs,
                       'Learning_Rate':learning_rate, 
                       'Device':str(device), 
                       'Criterion':'torch.nn.L1Loss(reduction=sum)',
                       'Optimizer':'torch.optim.Adam(model.parameters(), lr=learning_rate)',
                       'abnormal_data_index':list(dt_provider['abnormal_data_index'].astype(str).values),
                       'normal_data_train_index':list(dt_provider['normal_data_train_index'].astype(str).values),
                       'normal_data_test_index':list(dt_provider['normal_data_test_index'].astype(str).values),
                       'grey_data_index':list(dt_provider['grey_data_index'].astype(str).values),
                       'grey_data_train_index':list(dt_provider['grey_data_train_index'].astype(str).values),
                       'train_data_index':list(dt_provider['train_data_index'].astype(str).values),
                       'Number_featuers':dataframe.shape[1]-1,
                       'columns':list(dataframe.columns.values),
                       'Time_Window':str(Time_Window).replace(':','_'),
                       'Epochs_Results':[],
                       'null_index':list(dt_provider['null_index'].astype(str).values)}




    dir_path = '/home/seyedkazemi/dataset/Marconi100-Cooked/AE/LSTM-AE-5-Month/LSTM-AE_Results_SqL'+str(sequence_length)+'-ftur'+str(dataframe.shape[1]-1)+'grey_data_range_'+str(grey_data_range[0]).zfill(2)+'_'+str(grey_data_range[1]).zfill(2)+'-f'+str(grey_data_train_threshold).zfill(2)+'/Train_'+str(start_train.split(' ')[0])+'_to_'+str(stop_train.split(' ')[0])+'/'+dir_name+'/'
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)

    # save the scaler
    dump(dt_provider['fit_scaler_fun'], open(dir_path+'scaler.pkl', 'wb'))
    display(dt_provider)


    for epoch in range(1, epochs+1):
        model.train()
        train_loss = lstm_train_loop(dt_provider['train_dataloader'], model, criterion, optimizer,device=device)
        model.eval()
        test_loss = lstm_test_loop(dt_provider['test_dataloader'], model, criterion,device=device)
        test_loss_abnormal = lstm_test_loop(dt_provider['abnormal_dataloader'], model, criterion,device=device)    
        restults_complete['Epochs_Results'].append({'epoch':epoch,'TrainLoss':train_loss,'TestLoss':test_loss,'TestAbnormalLoss':test_loss_abnormal})

        if epoch%10==0:
            print(f"{epoch} Train loss: {train_loss:>7f} and Test loss: {test_loss:>7f} and Abnormal Test loss: {test_loss_abnormal:>7f}")


        if epoch%10==0:
            model_name = mohsenutils.file_name_creator(path=dir_path, file_name='model_epoch_'+str(epoch)+'.pt', log=False)
            torch.save(model, dir_path+model_name)    

    output_file_name = mohsenutils.file_name_creator(path=dir_path, file_name='output.txt', log=False)

    with open(dir_path+output_file_name, 'w') as outfile:  
        outfile.write(str(summary(model.float(),(1,sequence_length,dataframe.shape[1]-1))))
        outfile.write('#+#')
        json.dump(restults_complete, outfile)
        outfile.close()



    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # INFERENCE RESULTS    
    dataframe_scl = pd.DataFrame(dt_provider['fit_scaler_fun'].transform(dataframe.drop('sum_flags', axis=1)), columns=dataframe.drop('sum_flags', axis=1).columns)
    dataframe_scl.index = dataframe.index
    all_dataset = autoencoder_helper.LSTM_AE_dataset(dataframe=dataframe_scl, Time_Window=Time_Window, Index_lst=dataframe_scl.iloc[37:,:].index)

    inference_results = pd.DataFrame(columns=['Error'])

    for dtl_idx, X in enumerate(DataLoader(all_dataset)):
        model.eval()
        dt, cols, idx = X
        x = torch.transpose(dt,1,2).to(device).float()
        if dtl_idx == 0 :
            print(f'Input X size is :{x.size()}\nInput X[0]is:\n\t{x}')
        pred = model(x)
        loss = criterion(pred, x)
        inference_results.loc[idx[-1][0],'Error']=loss.item()


    inference_results.index.name = 'timestamp'
    inference_results.index = pd.to_datetime(inference_results.index)
    inference_results['Label'] =''
    inference_results.loc[dt_provider['normal_data_train_index'],'Label'] = 'train'
    inference_results.loc[dt_provider['normal_data_test_index'],'Label'] = 'test'
    inference_results.loc[dt_provider['abnormal_data_index'],'Label'] = 'abnormal'

    inference_results.loc[dt_provider['grey_data_index'],'Label'] = 'grey'
    inference_results.loc[dt_provider['grey_data_train_index'],'Label'] = 'grey_train'


    inference_results = mohsenutils.mrg(None,'inner',dataframe[['sum_flags']], inference_results)
    inference_results.to_csv(dir_path+'Inference_Results.csv') 

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
############################
#      Results Helper      ###############################################################
############################     
import os, sys
sys.path.append('/home/seyedkazemi/codes/mskhelper/')
import pandas as pd
import seaborn as sns
import numpy as np
import mohsenutils, function_set
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, plot_confusion_matrix
pd.options.display.float_format = "{:,.2f}".format


def add_label_grey_grey_train(df, grey_data_range_train_threshold, verbose=False):
    
    verboseprint = print if verbose else lambda *a, **k: None
    
    verboseprint()
    verboseprint(50*'+-')
    verboseprint()
    verboseprint('Unique Labels Before the Grey', df.Label.unique())
    
    grey_index = df.query('@grey_data_range_train_threshold[0] <= sum_flags <= @grey_data_range_train_threshold[1]').index
    grey_train_index = df.query('@grey_data_range_train_threshold[0] <= sum_flags <= @grey_data_range_train_threshold[2]').index
    verboseprint('Unique Sum Flags of Grey: ',np.sort(df.loc[grey_index,:].sum_flags.unique()))
    df.loc[grey_index,'Label']='grey'
    verboseprint('Unique Labels After the Grey',df.Label.unique())
    df.loc[grey_train_index,'Label']='grey_train'
    verboseprint('Unique Sum Flags Grey-Train: ',np.sort(df.query("Label == 'grey_train'").sum_flags.unique()))
    verboseprint('Unique Sum Flags Train: ',np.sort(df.query("Label == 'train'").sum_flags.unique()))
    verboseprint('Unique Labels After the Grey-Train',df.Label.unique())
    verboseprint()
    verboseprint(50*'+-')
    verboseprint()
    

    
def apply_set_fun(dataframes, verbose=False):
    verboseprint = print if verbose else lambda *a, **k: None
    verbosedisplay = display if verbose else lambda *a, **k: None
    for df in dataframes:
        verboseprint()
        verboseprint(30*'+-~')
        verboseprint()
        df.sort_index(inplace=True)
        df.drop(index = pd.to_datetime(df.index[0]), inplace=True) # Drop first Row scince there is no label in the firts row of the LSTM
        
        verbosedisplay(df)
        verboseprint('Unique Labels:', df['Label'].unique())
        verboseprint("abnormal", np.sort(df.query('Label=="abnormal"')['sum_flags'].unique()))
        verboseprint("train", np.sort(df.query('Label=="train"')['sum_flags'].unique()))
        verboseprint("test" ,np.sort(df.query('Label=="test"')['sum_flags'].unique()))
        verboseprint("Null",np.sort(df[df.Label.isnull()]['sum_flags'].unique()))
        verboseprint()
        verboseprint(30*'+-~')
        verboseprint()

        
        
def classifier_flag(row, *grey_data_range_train_threshold):
#     print(row)
    if row['sum_flags']>grey_data_range_train_threshold[1]:
        clss = 1
    elif row['sum_flags']<grey_data_range_train_threshold[0]:
        clss = 0
    else:
        clss = 0.5
    return clss




def classifier_error(row, train_error_threshold):
    if row['Error']>train_error_threshold:
        clss = 1
    else:
        clss = 0 
    return clss        
    
def process_error_AE(df, grey_data_range_train_threshold, train_error_threshold_quantile=None, train_error_threshold=None, verbose=False):
    
    assert (train_error_threshold_quantile != None) ^ (train_error_threshold != None), 'Please check the train_error_threshold_quantile and train_error_threshold one of these should be not None'
    
    verboseprint = print if verbose else lambda *a, **k: None
    
#     display(df)
    
    df_train = df.query("Label == 'train' or Label == 'grey_train'")
    df_test = df.query("Label == 'test'")
    df_abnormal = df.query("Label == 'abnormal'")
    df_grey = df.query("Label == 'grey'")

    # train_error_threshold = 3*df_train[['Error']].std().values
    # train_error_threshold = 5*df_train[['Error']].std().values
    
    if train_error_threshold == None:
        train_error_threshold = df_train[['Error']].quantile(train_error_threshold_quantile).values
    
        
    
    
#     train_error_threshold = [2]
    verboseprint('Error Thresholds:', train_error_threshold)
    if verbose:
        fig, axl = plt.subplots(1,7,figsize=(5*7,5))
        sns.histplot(df_train['sum_flags'], ax= axl[0])
        sns.histplot(df_test['sum_flags'], ax= axl[1])
        sns.histplot(df_abnormal['sum_flags'], ax= axl[2])
        sns.histplot(df_grey['sum_flags'], ax= axl[3])

        sns.histplot(df_train.Error, ax= axl[4])
        sns.histplot(df_test.Error, ax= axl[5])
        sns.histplot(df_abnormal.Error, ax= axl[6])

        axl[0].set_title('Train $\sum{}_{}Flags$')
        axl[1].set_title('Test $\sum{}_{}Flags$')
        axl[2].set_title('Abnormal $\sum{}_{}Flags$')
        axl[3].set_title('Grey $\sum{}_{}Flags$')

        axl[4].set_title('Train Error')
        axl[5].set_title('Test Error')
        axl[6].set_title('Abnormal Error')
    
    
    error_des = df_train[['Error']].describe()
    
    verboseprint('Error Describe:', error_des)
    
    df['class_flags'] = df[['sum_flags']].apply(classifier_flag, args=grey_data_range_train_threshold, axis=1) 
    df['class_error'] = df[['Error']].apply(classifier_error, args=train_error_threshold, axis=1) 
    return df, df_train, df_test, df_abnormal, df_grey, train_error_threshold, error_des


def acc_f1_results(dataframes_dic, verbose=False):
    verboseprint = print if verbose else lambda *a, **k: None
    
    metric_results = pd.DataFrame(columns=["grey_data_range_train_threshold","Acc. Test Normal", "Acc. Abnormal", "Av. Acc.", "Av. F1-score", "TN", "FP", "FN", "TP"])
    
    for i, df_dic in enumerate(dataframes_dic):
        
        acc_test_normal = accuracy_score(df_dic['df'].query("Label=='test'")['class_flags'],df_dic['df'].query("Label=='test'")['class_error'])
        acc_abnormal = accuracy_score(df_dic['df'].query("Label=='abnormal'")['class_flags'],df_dic['df'].query("Label=='abnormal'")['class_error'])
        acc_mean = accuracy_score(df_dic['df'].query("Label=='test'or Label=='abnormal'")['class_flags'],df_dic['df'].query("Label=='test'or Label=='abnormal'")['class_error'])
        f1_mean = f1_score(df_dic['df'].query("Label=='test'or Label=='abnormal'")['class_flags'],df_dic['df'].query("Label=='test'or Label=='abnormal'")['class_error'])
        TN, FP, FN, TP = function_set.conf_mat(df_dic['df'].query("Label=='test'or Label=='abnormal'")['class_flags'],df_dic['df'].query("Label=='test'or Label=='abnormal'")['class_error'])
        
        verboseprint(np.round(acc_test_normal, 2),np.round(acc_abnormal, 2),np.round(acc_mean, 2),np.round(f1_mean, 2))
        metric_results.loc[i,["grey_data_range_train_threshold","Acc. Test Normal", "Acc. Abnormal", "Av. Acc.", "Av. F1-score", "TN", "FP", "FN", "TP"]] = \
        [str(df_dic["grey_data_range_train_threshold"]), np.round(acc_test_normal, 2), np.round(acc_abnormal, 2), np.round(acc_mean, 2), np.round(f1_mean, 2), TN, FP, FN, TP]

    return metric_results



def AE_abnormal_grey(dataframes):    
    AE_abnormal_grey = pd.DataFrame(columns=['# Grey Train Dt Exld','Abnrml AE % of Train Exld','# Grey Train Dt Inld','Abnrml AE % of Train Inld'])

    for i in range(6):
        AE_abnormal_grey.loc[i,['# Grey Train Dt Exld','Abnrml AE % of Train Exld','# Grey Train Dt Inld','Abnrml AE % of Train Inld']]=\
        [dataframes[i].query("Label=='grey'").shape[0],
         dataframes[i].query("Label=='grey'")['class_error'].mean(),
         dataframes[i][((dataframes[i]['sum_flags']>0) & (dataframes[i]['sum_flags']<25+1))].shape[0],
         dataframes[i][((dataframes[i]['sum_flags']>0) & (dataframes[i]['sum_flags']<25+1))]['class_error'].mean()]


    return AE_abnormal_grey


def percentage_Class_One(dataframes):
    percentage_class1 =  pd.DataFrame(columns=['%Flags Abnormal','%Flags Normal','%Flags Grey','%AE Abnormal'])  
    
    for i, dataframe in enumerate(dataframes):
        percentage_class1.loc[i, ['%Flags Abnormal','%Flags Normal','%Flags Grey','%AE Abnormal']] =\
            [100*len(dataframe.query("class_flags==1"))/len(dataframe),
             100*len(dataframe.query("class_flags==0"))/len(dataframe),
             100*len(dataframe.query("class_flags==0.5"))/len(dataframe),
             100*len(dataframe.query("class_error==1"))/len(dataframe)]
    return percentage_class1


def compare(row, col1, col2):
#     print(row[col1], row[col2])
    if row[col1]==0.5:
#         print('yes')
        return np.nan
    else:
        if row[col1] == row[col2]:
#             print("OK")
            return 0 
        else:
#             print('Error')
            return 1


def add_label(df, grey_data_range_train_threshold, start_train, stop_train, verbose=False):
    
    verboseprint = print if verbose else lambda *a, **k: None
    
    verboseprint()
    verboseprint(50*'+-')
    verboseprint('Labels:', df[['Label']].value_counts(), '\nNull:',df['Label'].isnull().sum())
    verboseprint()
    verboseprint('Unique Labels Before the Grey', df.Label.unique())
    df.loc[start_train:stop_train, ['Label']] = 'Train_Month_'+df.loc[start_train:stop_train, ['Label']]
    verboseprint('Labels:', df[['Label']].value_counts(), '\nNull:',df['Label'].isnull().sum())
    not_null_index = df[~df['Label'].isnull()].index
    grey_index = df.query('@grey_data_range_train_threshold[0] <= sum_flags <= @grey_data_range_train_threshold[1]').index
    grey_train_index = df.query('@grey_data_range_train_threshold[0] <= sum_flags <= @grey_data_range_train_threshold[2]').index
    abnormal_index = df.query('sum_flags > @grey_data_range_train_threshold[1]').index
    normal_index =   df.query('sum_flags < @grey_data_range_train_threshold[0]').index
    grey_index = pd.to_datetime([idx for idx in grey_index if idx not in not_null_index])
    grey_train_index = pd.to_datetime([idx for idx in grey_train_index if idx not in not_null_index])
    abnormal_index = pd.to_datetime([idx for idx in abnormal_index if idx not in not_null_index])
    normal_index = pd.to_datetime([idx for idx in normal_index if idx not in not_null_index])
#     verboseprint('\n\ngrey_index:', grey_index, '\n\ngrey_train_index:',grey_train_index, '\n\nabnormal_index:', abnormal_index, '\n\nnormal_index:', normal_index)
    verboseprint('Unique Sum Flags of Grey: ',np.sort(df.loc[grey_index,:].sum_flags.unique()))
    df.loc[grey_index,'Label']='grey'
    verboseprint('Unique Labels After the Grey',df.Label.unique())
    df.loc[grey_train_index,'Label']='grey_train'
    verboseprint('Unique Sum Flags Grey-Train: ',np.sort(df.query("Label == 'grey_train'").sum_flags.unique()))
    verboseprint('Unique Sum Flags Train: ',np.sort(df.query("Label == 'train'").sum_flags.unique()))
    verboseprint('Unique Labels After the Grey-Train',df.Label.unique())
    df.loc[abnormal_index,'Label']='abnormal'
    df.loc[normal_index,'Label']='normal'
    verboseprint('Labels:', df[['Label']].value_counts(), '\nNull:',df['Label'].isnull().sum())
    verboseprint()
    verboseprint(50*'+-')
    verboseprint()
    
    
    
    
    
    
    
    









##################################
# inference_prediction_dataframe #
##################################




import os, sys, imp
sys.path.append('/home/seyedkazemi/codes/mskhelper/')
sys.path.append('/home/seyedkazemi/codes/Marconi100/DNN/')
import pandas as pd, numpy as np
import datetime, pytz
import data_provider, mohsenutils, m100_preprocessing_helper, flag_helper
import imp 
from  matplotlib import pyplot as plt
import torch 
import torch.nn as nn
import torch.functional as f
from torch.utils.data import DataLoader, Dataset
# from torchsummary import summary
from torchinfo import summary

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import autoencoder_helper
from pickle import load
import json 

imp.reload(data_provider)
imp.reload(mohsenutils)
imp.reload(m100_preprocessing_helper)
imp.reload(flag_helper)
imp.reload(autoencoder_helper)

import random





def inference_prediction_dataframe(save_results_dir_path, model_dir_path, pred_dataframe_file_name, inference_file_name):
    SEED = 1234
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    ################
    #  Parameters  #
    ################
    output = model_dir_path +                  'output.txt'
    scaler_path = model_dir_path +             'scaler.pkl'
    model_path = model_dir_path +              'model_epoch_100.pt'
    Inference_Results_file = model_dir_path + 'Inference_Results.csv'

    Time_Window        = datetime.timedelta(minutes=20)
    Sampling_Frequency = datetime.timedelta(minutes=10)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    criterion =      torch.nn.L1Loss(reduction='sum').to(device)
#     criterion_none = torch.nn.L1Loss(reduction='none').to(device)

    inference_results = pd.DataFrame(columns=['Error'])
    pred_dataframe = pd.DataFrame()



    if not os.path.isdir(save_results_dir_path):
        os.makedirs(save_results_dir_path)

    with open(output) as text:
        txt = text.read()
    restults_complete = json.loads(txt.split('#+#')[1])

    print()
    print(restults_complete.keys())
    print()
    print(f'Inference_Results_file: {Inference_Results_file}')
    print()
    print(f'scaler_path: {scaler_path}')
    print()
    print(f'output: {output}')
    print()
    print(f'model_path: {model_path}')
    print()


    flags_data_kwargs = {'rack':restults_complete['Rack'], 
                         'ambient_dt':True, 'pcie_dt':True, 'fsnd_dt':True, 'psnd_dt':True, 'gpu0_dt':True, 'gpu1_dt':True, 'gpu3_dt':True, 'gpu4_dt':True, 
                         'cpu0_dt':True, 'cpu1_dt':True, 'Tot_ict_dt':True, 'Tot_qpompe_dt':True, 'Tot_chiller_dt':True,
                         'RDHX_dt':True, 'Supply_Air_Temperature_dt':True, 'Return_Air_Temperature_dt':True, 
                         'Free_Cooling_Valve_Open_Position_dt':True, 'Tot_cdz_dt':True, 'FScrac_dt':True, 'Free_Cooling_Status_dt':True, 'Compressor_Utilization_dt':True, 'temp_dt':True}
    dataframe = autoencoder_helper.flags_data_new(**flags_data_kwargs)
    


    print(25*'+-+')
    print('Orginal DataFrame Shape: ', dataframe.shape)
    print()
    print(f'Sum of null {dataframe.isnull().sum()[dataframe.isnull().sum()>0]}')
    print()
    print(f'Sum of null {dataframe.T.isnull().sum()[dataframe.T.isnull().sum()>0]}')
    print()
    print(f'Sum of sum of null {dataframe.isnull().sum().sum()}')
    dataframe = dataframe.interpolate()
    print()
    print(f'Sum of null after interpolation: {dataframe.isnull().sum()[dataframe.isnull().sum()>0]}')
    print()
    display(dataframe)

    model = torch.load(model_path).to(device)
    display(model)

    # load the scaler
    scaler = load(open(scaler_path, 'rb'))
    dataframe_scl = pd.DataFrame(scaler.transform(dataframe.drop('sum_flags', axis=1)), columns=dataframe.drop('sum_flags', axis=1).columns)
    dataframe_scl.index = dataframe.index

    index_list_null_removed, null_index = autoencoder_helper.cancel_index_with_null(dataframe = dataframe.drop('sum_flags', axis=1), Time_Window=Time_Window, total_null_threshold=12, row_null_threshold=6)
    print()
    print('index_list_null_removed :', index_list_null_removed)
    print()
    print('null_index :', null_index)
    # Remove first samples related to the Time Window:
    index_list_null_removed = index_list_null_removed[index_list_null_removed>dataframe_scl.index[37]]
    print('index_list_null_removed :',index_list_null_removed)

    all_dataset = autoencoder_helper.LSTM_AE_dataset(dataframe=dataframe_scl, Time_Window=Time_Window, Index_lst=index_list_null_removed)



    for X in DataLoader(all_dataset):
        model.eval()
        dt, cols, idx = X
        cols = [col[0] for col in cols]
        idx = [i[0] for i in idx]

        x = torch.transpose(dt,1,2).to(device).float()
        pred = model(x)
        df_pred = mohsenutils.df_shift_merge(dataframe=pd.DataFrame(pred.detach().to('cpu').numpy()[0], index=idx, columns=cols), shift_range=int(Time_Window/Sampling_Frequency)).iloc[[-1]]
        pred_dataframe = pd.concat([pred_dataframe, df_pred])
        loss = criterion(pred, x)
        inference_results.loc[idx[-1],'Error'] = loss.item()


    # Add index which cancelled from inference_results due to the null of the dataframe.  
    print()
    print('Shape of the Null cancelled data: ',inference_results.shape)
    inference_results.index = pd.to_datetime(inference_results.index)
    inference_results.sort_index(inplace=True)
    inference_results = inference_results.merge(pd.DataFrame(index=pd.date_range(start=inference_results.index[0], end=inference_results.index[-1], freq='10T')), how='outer', right_index=True, left_index=True)
    inference_results.sort_index(inplace=True)
    print()
    print('Shape of data after adding the cancelled rows: ', inference_results.shape)
    print()
    print('Null of inference_results Error due to the Null of the dataframe', inference_results['Error'].isnull().sum())


    inference_results.index.name = 'timestamp'
    inference_results.index = pd.to_datetime(inference_results.index)
    inference_results['Label'] =''
    inference_results.loc[restults_complete['abnormal_data_index'],'Label'] = 'abnormal'
    inference_results.loc[restults_complete['normal_data_train_index'],'Label'] = 'train'
    inference_results.loc[restults_complete['normal_data_test_index'],'Label'] = 'normal_test'
    inference_results.loc[restults_complete['grey_data_index'],'Label'] = 'grey'
    inference_results.loc[restults_complete['grey_data_train_index'],'Label'] = 'grey_train'
    inference_results = mohsenutils.mrg(None,'inner',dataframe[['sum_flags']], inference_results)

    inference_results.to_csv(save_results_dir_path+inference_file_name)
    pred_dataframe.to_csv(save_results_dir_path+pred_dataframe_file_name)
    display(inference_results)
    print('Done!')

    
    
    
#######################################    
# Error loaction and Error Loacation  ############################################################   
#######################################    

    
def L1loss_row_sum(prediction_df, ground_truth_df):
    mohsenutils.check_same_columns(prediction_df, ground_truth_df)    
    assert prediction_df.shape == ground_truth_df.shape, 'The shape of the prediction_df, ground_truth_df should be same !!!' + str(prediction_df.shape) + str(ground_truth_df.shape)
    l1_loss = np.abs(prediction_df - ground_truth_df).sum(axis=1)
    l1_loss = pd.DataFrame({'Sum':l1_loss})
    return l1_loss


def L1loss(prediction_df, ground_truth_df):
    mohsenutils.check_same_columns(prediction_df, ground_truth_df)    
    assert prediction_df.shape == ground_truth_df.shape, 'The shape of the prediction_df, ground_truth_df should be same !!!'+ str(prediction_df.shape) + str(ground_truth_df.shape)
    return np.abs(mohsenutils.diff_pd_dataframe(prediction_df, ground_truth_df))


def Error_1(prediction_df, ground_truth_df):
    """
    Error =  |prediction_df - ground_truth_df| / max(|prediction_df|, |ground_truth_df|)
    
    prediction_df=NaN, or/and ground_truth_df=NaN  ==> Error=NaN
    prediction_df=0, and ground_truth_df=0  ==> Error=0
    """
    
    mohsenutils.check_same_columns(prediction_df, ground_truth_df)
    assert prediction_df.shape == ground_truth_df.shape, 'The shape of the prediction_df, ground_truth_df should be same !!!'+ str(prediction_df.shape) + str(ground_truth_df.shape)
    
    df_max = np.abs(prediction_df).where(np.abs(prediction_df)>np.abs(ground_truth_df), np.abs(ground_truth_df))
    df_max = df_max.where(~prediction_df.isnull(), np.nan)
    
    error = np.abs(mohsenutils.diff_pd_dataframe(prediction_df, ground_truth_df))/df_max
    error[(prediction_df==0)&(ground_truth_df==0)] = 0

    return error

def Error_2(prediction_df, ground_truth_df):
    
    """
    Error =  |prediction_df - ground_truth_df| / max(|ground_truth_df|)
    """
    mohsenutils.check_same_columns(prediction_df, ground_truth_df)    
    assert prediction_df.shape == ground_truth_df.shape, 'The shape of the prediction_df, ground_truth_df should be same !!!'+ str(prediction_df.shape) + str(ground_truth_df.shape)
    return np.abs(mohsenutils.diff_pd_dataframe(prediction_df, ground_truth_df)/np.abs(ground_truth_df).max())

 
def error_location_time_sensors(Loss_function, prediction_df, ground_truth_df, L1loss_row_sum_range, loss_item_range_greater_than):
    """
    return dictionary with timestamp and names of the columns that has high loss.
    """
    l1loss_row = L1loss_row_sum(prediction_df, ground_truth_df)
    loss_df = Loss_function(prediction_df, ground_truth_df)
    
    if L1loss_row_sum_range[0] != None:
        l1loss_row = l1loss_row[(l1loss_row['Sum'] >= L1loss_row_sum_range[0])]
    
    if L1loss_row_sum_range[1] != None:
        l1loss_row = l1loss_row[(l1loss_row['Sum'] <= L1loss_row_sum_range[1])]
        
    loss_df = loss_df.loc[l1loss_row.index,:]
    
    
    final_results = {}
    
    def rows_cols_greater_than(row, loss_item_range_greater_than):
        if row[row > loss_item_range_greater_than].shape[0] > 0:
            return row.name, row[row > loss_item_range_greater_than].to_dict() 
        

    for row in loss_df.iterrows():
        rslt = rows_cols_greater_than(row=row[1], loss_item_range_greater_than=loss_item_range_greater_than)
        if rslt != None:
            index, dt = rslt
            final_results[str(index)] = dt
            
            
    return final_results




import sys
sys.path.append('/home/seyedkazemi/codes/mskhelper/')
sys.path.append('/home/seyedkazemi/codes/Marconi100/DNN/')
import pandas as pd, numpy as np
from sklearn.preprocessing import StandardScaler
from pickle import load
import mohsenutils, autoencoder_helper
import imp
imp.reload(mohsenutils)
imp.reload(autoencoder_helper)

################
#  Parameters  #
################

def error_physical_units(Loss_function, scaler_path, prediction_scl_dataframe_file_path):
    # load the scaler
    scaler = load(open(scaler_path, 'rb'))
    
    prediction_scl = pd.read_csv(prediction_scl_dataframe_file_path, index_col='Unnamed: 0', parse_dates=True, date_parser=pd.to_datetime)
    prediction_scl.index.name='timestamp'
    prediction_scl.sort_index(inplace=True)
    prediction_scl.sort_index(inplace=True)

    prediction_Rescl = mohsenutils.mrg(None,'inner',
                                       pd.DataFrame(scaler.inverse_transform(prediction_scl.iloc[:, :242]), 
                                                    columns=prediction_scl.iloc[:, :242].columns, 
                                                    index=prediction_scl.iloc[:, :242].index),
                                       pd.DataFrame(scaler.inverse_transform(prediction_scl.iloc[:, 242:]), 
                                                    columns=prediction_scl.iloc[:, 242:].columns, 
                                                    index=prediction_scl.iloc[:, 242:].index))

    prediction_Rescl.sort_index(inplace=True)
    
    display('Prediction scaled:', prediction_scl)
    display('Prediction Rescaled:',prediction_Rescl)
    mohsenutils.check_same_columns(prediction_scl, prediction_Rescl)

    
    flags_data_kwargs = {'rack':'205', 
                         'ambient_dt':True, 'pcie_dt':True, 'fsnd_dt':True, 'psnd_dt':True, 'gpu0_dt':True, 'gpu1_dt':True, 'gpu3_dt':True, 'gpu4_dt':True, 
                         'cpu0_dt':True, 'cpu1_dt':True, 'Tot_ict_dt':True, 'Tot_qpompe_dt':True, 'Tot_chiller_dt':True,
                         'RDHX_dt':True, 'Supply_Air_Temperature_dt':True, 'Return_Air_Temperature_dt':True, 
                         'Free_Cooling_Valve_Open_Position_dt':True, 'Tot_cdz_dt':True, 'FScrac_dt':True, 'Free_Cooling_Status_dt':True, 'Compressor_Utilization_dt':True, 'temp_dt':True}
    dataframe = autoencoder_helper.flags_data_new(**flags_data_kwargs)


    print(25*'+-+')
    print('Orginal DataFrame Shape: ', dataframe.shape)
    print()
    print(f'Sum of null {dataframe.isnull().sum()[dataframe.isnull().sum()>0]}')
    print()
    print(f'Sum of null {dataframe.T.isnull().sum()[dataframe.T.isnull().sum()>0]}')
    print()
    print(f'Sum of sum of null {dataframe.isnull().sum().sum()}')
    dataframe = dataframe.interpolate()
    print()
    print(f'Sum of null after interpolation: {dataframe.isnull().sum()[dataframe.isnull().sum()>0]}')
    print()
    display(dataframe)



    # index_list_null_removed, null_index = autoencoder_helper.cancel_index_with_null(dataframe = dataframe.drop('sum_flags', axis=1), Time_Window=Time_Window, total_null_threshold=12, row_null_threshold=6)
    # print()
    # print('index_list_null_removed :', index_list_null_removed)
    # print()
    # print('null_index :', null_index)
    # # Remove first samples related to the Time Window:
    # index_list_null_removed = index_list_null_removed[index_list_null_removed>dataframe_scl.index[37]]
    # print('index_list_null_removed :',index_list_null_removed)



    ground_truth_df = mohsenutils.df_shift_merge(
        dataframe=dataframe.drop('sum_flags',axis=1), 
        shift_range=2)



    prediction_Rescl = mohsenutils.mrg(None, 'outer', prediction_Rescl,pd.DataFrame(index=ground_truth_df.index))
    ground_truth_df =  mohsenutils.mrg(None, 'outer', ground_truth_df,pd.DataFrame(index=prediction_Rescl.index))

    error_phy_units = Loss_function(prediction_df=prediction_Rescl , ground_truth_df=ground_truth_df)
    return error_phy_units






def save_results_error_location_time_sensors(model_dir_path, L1loss_row_sum_range, loss_item_range_greater_than, Loss_function, output_file_name='Error_location_Inference_to_2021_08_17_23_50.txt'):
    
    import os, sys, imp, datetime, pytz, imp
    sys.path.append('/home/seyedkazemi/codes/mskhelper/')
    sys.path.append('/home/seyedkazemi/codes/Marconi100/DNN/')
    import pandas as pd, numpy as np
    import data_provider, mohsenutils, m100_preprocessing_helper, flag_helper
    from  matplotlib import pyplot as plt
    import torch 
    import torch.nn as nn
    import torch.functional as f
    from torch.utils.data import DataLoader, Dataset
    from torchsummary import summary
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    import autoencoder_helper
    from pickle import load
    import json 

    imp.reload(data_provider)
    imp.reload(mohsenutils)
    imp.reload(m100_preprocessing_helper)
    imp.reload(flag_helper)
    imp.reload(autoencoder_helper)

    import random
    SEED = 1234
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    
    save_results_dir_path =    model_dir_path + '/Error_Location/'
    pred_dataframe_file_name = 'prediction_Inference_to_2021_08_17_23_50.csv'
    


    output =                 model_dir_path + 'output.txt'
    scaler_path =            model_dir_path + 'scaler.pkl'


    Time_Window = datetime.timedelta(minutes=20)
    Sampling_Frequency = datetime.timedelta(minutes=10)


    with open(output) as text:
        txt = text.read()
    restults_complete = json.loads(txt.split('#+#')[1])

    print()
    print(restults_complete.keys())
    print()
    print(f'scaler_path: {scaler_path}')
    print()
    print(f'output: {output}')
    print()


    flags_data_kwargs = {'rack':restults_complete['Rack'], 
                         'ambient_dt':True, 'pcie_dt':True, 'fsnd_dt':True, 'psnd_dt':True, 'gpu0_dt':True, 'gpu1_dt':True, 'gpu3_dt':True, 'gpu4_dt':True, 
                         'cpu0_dt':True, 'cpu1_dt':True, 'Tot_ict_dt':True, 'Tot_qpompe_dt':True, 'Tot_chiller_dt':True,
                         'RDHX_dt':True, 'Supply_Air_Temperature_dt':True, 'Return_Air_Temperature_dt':True, 
                         'Free_Cooling_Valve_Open_Position_dt':True, 'Tot_cdz_dt':True, 'FScrac_dt':True, 'Free_Cooling_Status_dt':True, 'Compressor_Utilization_dt':True, 'temp_dt':True}
    dataframe = autoencoder_helper.flags_data_new(**flags_data_kwargs)


    print(25*'+-+')
    print('Orginal DataFrame Shape: ', dataframe.shape)
    print()
    print(f'Sum of null {dataframe.isnull().sum()[dataframe.isnull().sum()>0]}')
    print()
    print(f'Sum of null {dataframe.T.isnull().sum()[dataframe.T.isnull().sum()>0]}')
    print()
    print(f'Sum of sum of null {dataframe.isnull().sum().sum()}')
    dataframe = dataframe.interpolate()
    print()
    print(f'Sum of null after interpolation: {dataframe.isnull().sum()[dataframe.isnull().sum()>0]}')
    print()
    display(dataframe)


    # load the scaler
    scaler = load(open(scaler_path, 'rb'))
    dataframe_scl = pd.DataFrame(scaler.transform(dataframe.drop('sum_flags', axis=1)), columns=dataframe.drop('sum_flags', axis=1).columns)
    dataframe_scl.index = dataframe.index
    dataframe_scl_shift_merge = mohsenutils.df_shift_merge(dataframe=dataframe_scl, shift_range=2)
    display(dataframe_scl_shift_merge)


    # index_list_null_removed, null_index = autoencoder_helper.cancel_index_with_null(dataframe = dataframe.drop('sum_flags', axis=1), Time_Window=Time_Window, total_null_threshold=12, row_null_threshold=6)
    # print()
    # print('index_list_null_removed :', index_list_null_removed)
    # print()
    # print('null_index :', null_index)
    # # Remove first samples related to the Time Window:
    # index_list_null_removed = index_list_null_removed[index_list_null_removed>dataframe_scl.index[37]]
    # print('index_list_null_removed :',index_list_null_removed)





    prediction = pd.read_csv(save_results_dir_path+pred_dataframe_file_name, index_col='Unnamed: 0', parse_dates=True, date_parser=pd.to_datetime)
    prediction.index.name='timestamp'
    prediction.sort_index(inplace=True)
    display(prediction)




    prediction = mohsenutils.mrg(None, 'outer', prediction, pd.DataFrame(index = dataframe_scl_shift_merge.index))
    dataframe_scl_shift_merge = mohsenutils.mrg(None, 'outer', dataframe_scl_shift_merge, pd.DataFrame(index = prediction.index))
    prediction, dataframe_scl_shift_merge = mohsenutils.same_index_range(prediction, dataframe_scl_shift_merge)

    display(prediction, dataframe_scl_shift_merge)


    final_results = autoencoder_helper.error_location_time_sensors(Loss_function=Loss_function,
                                                                   prediction_df=prediction,
                                                                   ground_truth_df=dataframe_scl_shift_merge, 
                                                                   L1loss_row_sum_range=L1loss_row_sum_range,
                                                                   loss_item_range_greater_than=loss_item_range_greater_than)

    with open(save_results_dir_path+output_file_name, 'w') as outfile:  
        outfile.write(str(final_results))
        outfile.close()


    print('Done!')
###################################
# End of Error and Error loaction ############################################################################
###################################







def error_location_AEandflgas(ae_error_location_txt_pth, falgs_csv_pth, timestamp, ae_recon_error_prec=0.1):
    
    with open(ae_error_location_txt_pth) as text:
        txt = text.read()

    txt = txt.replace("\'", "\"")
    Error_location = json.loads(txt)

    keys_rng = pd.to_datetime(list(Error_location.keys()))
    keys_rng = keys_rng.sort_values()



   
    error_location_ae_dt = pd.DataFrame.from_dict(Error_location[str(timestamp)], orient='index', columns=['Error_of_Reconstruction'])
    error_location_ae_dt = error_location_ae_dt.query('Error_of_Reconstruction>@ae_recon_error_prec')
    error_location_ae = set([i.split('_pst_1')[0] for i in list(error_location_ae_dt.index) ])

    # 1 means 100% error in reconstruction
    print('Autoencoder',25*'++')
    print('1 means 100% error in reconstruction')
    print(f'Number of the sensors with more than {ae_recon_error_prec}% AE reconstruction error : {len(error_location_ae)} out of 242')

    
    # Flags:
    kwargs = {'parse_dates':True, 'date_parser':pd.to_datetime, 'index_col':'timestamp'}
    flags99 = pd.read_csv(falgs_csv_pth,**kwargs)
    flags_error_location_dt = flags99.loc[timestamp,:]
    flags_error_location_dt = flags_error_location_dt[flags_error_location_dt>0]

    flags_error_location = set(flags_error_location_dt.index.values)
    print(f'Number of the sensors with raised flgas {len(flags_error_location)} out of 281')
    error_location_ae_dt.index.name='parameters'
    flags_error_location_dt.name='flags_value' 
    
    flags_error_location_dt = pd.DataFrame(flags_error_location_dt)
    flags_error_location_dt.index.name='flags' 
    
    error_location_ae = pd.DataFrame(error_location_ae,columns=['parameters']) 
    flags_error_location = pd.DataFrame(flags_error_location, columns=['flags'])
    
    
    lpt_AE = pd.read_csv('/home/seyedkazemi/codes/Marconi100/Results_Analysis/csv_files/LookupTabel_AE.csv', index_col='Index')
    lpt_flags = pd.read_csv('/home/seyedkazemi/codes/Marconi100/Results_Analysis/csv_files/LookupTabel_Flags.csv', index_col='Index')
    
    error_location_ae['Err_Location'] = error_location_ae[['parameters']].applymap(lambda param : lpt_AE[lpt_AE['dataframe_parameters']==param].Type.values[0])
    flags_error_location['Err_Location'] = flags_error_location[['flags']].applymap(lambda flg : lpt_flags[lpt_flags['Flags']==flg].Type.values[0])
    
    
    
    erl_ae =  pd.DataFrame(error_location_ae.Err_Location.value_counts())
    erl_ae.rename(columns={'Err_Location':'num_Err_Location_AE'}, inplace=True)
    erl_ae.index.name =  'Err_Location'

    erl_flags =  pd.DataFrame(flags_error_location.Err_Location.value_counts())
    erl_flags.rename(columns={'Err_Location':'num_Err_Location_flags'}, inplace=True)
    erl_flags.index.name =  'Err_Location'

    
    
    max_lpt_flags = pd.DataFrame(lpt_flags.Type.value_counts())
    max_lpt_flags.index.name = 'Err_Location'
    max_lpt_flags.rename(columns={'Type':'Type_max_flags'}, inplace=True)
    max_lpt_flags.reset_index(inplace=True)

    max_lpt_AE = pd.DataFrame(lpt_AE.Type.value_counts())
    max_lpt_AE.index.name = 'Err_Location'
    max_lpt_AE.rename(columns={'Type':'Type_max_AE'}, inplace=True)
    max_lpt_AE.reset_index(inplace=True)
    
    
    erl_ae = pd.merge(pd.merge(max_lpt_AE, max_lpt_flags, on='Err_Location',how='outer')
             , erl_ae, on='Err_Location',how='outer')
    
    erl_ae_flags = pd.merge(erl_ae, erl_flags, on='Err_Location',how='outer')

    erl_ae_flags = erl_ae_flags.fillna(0)
    erl_ae_flags['%AE'] = erl_ae_flags['num_Err_Location_AE']/erl_ae_flags['Type_max_AE']
    erl_ae_flags['%flags'] = erl_ae_flags['num_Err_Location_flags']/erl_ae_flags['Type_max_flags']
    
    return erl_ae_flags, error_location_ae, flags_error_location,  error_location_ae_dt, flags_error_location_dt