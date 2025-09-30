import pandas as pd
import numpy as np
import datetime
import random
import os, sys, imp
sys.path.append('/home/seyedkazemi/codes/HPCRoomModel/mskhelper/')
import mohsenutils
imp.reload(mohsenutils)



def node_selector(cluster, height=None, rack=None):
    """
    node_list = node_selector(cluster, height=None, rack=None)
    """
    assert cluster in ['A2', 'M100'], 'cluster is NOT correcte. either "A2" or "M100"'
    
    if cluster=='A2':
        node_list = list(np.loadtxt('/home/seyedkazemi/codes/HPCRoomModel/mskhelper/A2_node_list.csv', delimiter=',',dtype='str'))
        
        if not height==None:
            node_list = [node for node in node_list if 'c'+str(height) in node]
        
        if not rack==None:
            node_list = [node for node in node_list if 'r'+str(rack) in node]
        
        print(f'{len(node_list)} nodes is selected.')
        assert  len(node_list)>=4, 'Please check the rack and height !!!!!!!!!!'
        print('For example:',random.sample(node_list, k = 4))
        
    elif cluster=='M100':
        node_list = list(np.loadtxt('/home/seyedkazemi/codes/HPCRoomModel/mskhelper/M100_node_list.csv', delimiter=',',dtype='str'))
        node_list.remove('login05')
        if not height==None:
            node_list = [node for node in node_list if 'n'+str(height) in node]
        
        if not rack==None:
            node_list = [node for node in node_list if 'r'+str(rack) in node]
        
        print(f'{len(node_list)} nodes is selected.')
        assert  len(node_list)>=1, 'Please check the rack and height !!!!!!!!!!'
        try:
            print('For example:',random.sample(node_list, k = 4))
        except:
            print('For example:',random.sample(node_list, k = 1))
        
        
    else:
        print('Error in the name of the cluster')
    
    return node_list





def data_path(cluster, interpolation, data_type):
    '''
    path = data_path(cluster, interpolation, data_type)
    interpolation:{'after', 'before'}
    A2 ==> data_type:{'Inlet', 'Outlet', 'Power'}
    M100 ==> data_type:{'ambient', 'pcie', 'total_power'}
    '''
    assert cluster in ['A2', 'M100'], 'Error in cluster name!!!! it should be either "A2" or "M100"'
    
    if cluster == 'A2':
        assert interpolation in ['after', 'before'], 'interpolation is NOT correcte. "after" or "before"'
        assert data_type in ['Inlet', 'Outlet', 'Power'], 'data_type is NOT correcte'
        
        
        if interpolation == "after":
    
            if data_type == 'Inlet':
                pth = '/home/seyedkazemi/dataset/complete_dataset/roomM_BB_Inlet_Temp_1min_01_2019_to_01_2020_after_interpolate_sampeling_10T.csv'
            elif data_type == 'Outlet':
                pth = '/home/seyedkazemi/dataset/complete_dataset/roomM_Exit_Air_Temp_1min_01_2019_to_01_2020_after_interpolate_sampeling_10T.csv'
            elif data_type == 'Power':
                pth = '/home/seyedkazemi/dataset/complete_dataset/roomM_KNL_PS_Input_Power_1min_01_2019_to_01_2020_after_interpolate_sampeling_10T.csv'
            else:
                pth = 'Error'
        
        elif interpolation == "before":
        
        
            if data_type == 'Inlet':
                pth = '/home/seyedkazemi/dataset/complete_dataset/roomM_BB_Inlet_Temp_1min_01_2019_to_01_2020_with_out_interpolate_10T.csv'
            elif data_type == 'Outlet':
                pth = '/home/seyedkazemi/dataset/complete_dataset/roomM_Exit_Air_Temp_1min_01_2019_to_01_2020_with_out_interpolate_10T.csv'
            elif data_type == 'Power':
                pth = '/home/seyedkazemi/dataset/complete_dataset/roomM_KNL_PS_Input_Power_1min_01_2019_to_01_2020_with_out_interpolate_sampeling_10T.csv'
            else:
                pth = 'Error'
        
        else:
            pth = 'Error'
        
        
        
    elif cluster == 'M100':
        
        assert interpolation in ['after', 'before'], 'interpolation is NOT correcte. "after" or "before"'
        assert data_type in ['ambient', 'pcie', 'total_power'], 'data_type is NOT correcte, "ambient", "pcie", "total_power"'
        
        
        if interpolation == "before":
            if data_type == 'ambient':
                pth = '/home/seyedkazemi/dataset/complete_dataset/ambient_10T.csv'
            elif data_type == 'pcie':
                pth = '/home/seyedkazemi/dataset/complete_dataset/pcie_10T.csv'
            elif data_type == 'total_power':
                pth = '/home/seyedkazemi/dataset/complete_dataset/total_power_10T.csv'
            else:
                pth = 'Error'
        
        elif interpolation == "after":
            if data_type == 'ambient':
                pth = '/home/seyedkazemi/dataset/complete_dataset/'
            elif data_type == 'pcie':
                pth = '/home/seyedkazemi/dataset/complete_dataset/'
            elif data_type == 'total_power':
                pth = '/home/seyedkazemi/dataset/complete_dataset/'
            else:
                pth = 'Error'
        
        else:
            pth = 'Error'
        
        
        
    else:
        print('Error in cluster name!!!! it should be either "A2" or "M100"')
    
    print('Data Path: ', pth)    
    return pth






def data_read(data_path, cluster, sampling_rate=None, selected_node=None, nrows=None):
    """
    data = data_read(data_path, cluster, sampling_rate=None, selected_node=None, nrows=None)
    """
    
    assert cluster in ['A2', 'M100'], 'Error in cluster name!!!! it should be either "A2" or "M100"'
    
    if cluster == 'A2':
        if not selected_node == None:
            selected_node.append('index')
    
        data = pd.read_csv(data_path, parse_dates=True, date_parser=pd.to_datetime, index_col='index', usecols=selected_node, nrows=nrows)
    
    elif cluster == 'M100':
        
        if not selected_node == None:
            selected_node.append('timestamp')
        
        data = pd.read_csv(data_path, parse_dates=True, date_parser=pd.to_datetime, index_col='timestamp', usecols=selected_node, nrows=nrows)
        data.index = pd.to_datetime(data.index, utc=True)
        data = data.tz_convert('Europe/Rome')
    
    
    
    else:
        print('Error in cluster name!!!! it should be either "A2" or "M100"')
    

    
    if not sampling_rate==None:
        data = data.resample(sampling_rate).mean()
        
    print('Data shape:', data.shape)
    
    return data












def data_date_filter(data, start=None, end=None):

    '''
    data = data_date_filter(data, start='2019-06-01 00:00:00', end='2019-07-01 00:00:00')
    
    '''
    data = mohsenutils.data_index_conv(data)
    if not start==None:
        data = data.loc[data.index>=start,:]
    if not end==None:
        data = data.loc[data.index<end,:]
    print('Data shape after data_date_filter ', data.shape)
    data.index.name = 'timestamp'
    return data


def data_provider(cluster, interpolation, data_type, sampling_rate=None, nrows=None, height=None, rack=None, start=None, end=None):
    print('Data Type '+data_type)
    data = data_read(cluster=cluster, 
                     data_path=data_path(cluster=cluster, interpolation=interpolation, data_type=data_type), 
                     sampling_rate=sampling_rate, 
                     selected_node=node_selector(cluster=cluster, height=height, rack=rack),
                     nrows=nrows)
    data = data_date_filter(data=data, start=start, end=end)
    print('--'*30,'\n')
    return data


def _filter(data, **masks):
    
    '''
    This function can filter out the a pandas dataframe with condtion on different columns. 
    data, filter_rules, filter_rules_txt = _filter(data, column1=[0.2, 0.5], column2=[None, 1], ... ,columnN=[10, 50])
    '''
    
    filter_rules = []
    for col, _ in masks.items():
        assert col in data.columns, 'The name of columns in the masks list are NOT correct !!! Please check the '+str(col)
    
    
    for col, min_max in masks.items():
        
        if not min_max[0]== None:
            print(col,'>',min_max[0])
            filter_rules.append(str(col)+'>'+str(min_max[0]))
            mask = data[col] > min_max[0]
            data = data[mask]
        
        if not min_max[1]== None:
            print(col,'<',min_max[1])
            filter_rules.append(str(col)+'<'+str(min_max[1]))
            mask = data[col] < min_max[1]
            data = data[mask]
        
    
    filter_rules_txt = ''
    for rule in filter_rules:
        rule = rule.replace('<', 'less')
        rule = rule.replace('>', 'greater')
        rule = rule.replace('==', 'equals')
        filter_rules_txt += rule
    print(filter_rules_txt)    
    
    return data, filter_rules, filter_rules_txt





def item_cnclr(index_list, window, i):
    mask_min = index_list > index_list[i]-datetime.timedelta(**window)
    mask_max = index_list < index_list[i]+datetime.timedelta(**window)
    mask = ~ (mask_min & mask_max)
    index_list = index_list[mask]
    return index_list

def cnclr(index_list, window):
    remain = []
    while not len(index_list)==0:
        remain.append(index_list[0])
        index_list = item_cnclr(index_list, window, i=0)
    return remain


def slicer(data, index, window):
    '''
    data = slicer(data, index, window)
    '''
    min_mask = data.index > index - datetime.timedelta(**window)
    max_mask = data.index < index + datetime.timedelta(**window)
    mask = min_mask & max_mask
    return data[mask]


