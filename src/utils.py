import sys
sys.path.append("/home/seyedkazemi/codes/HPCRoomModel/mskhelper/")

import datetime as dt
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import os
import numpy as np
import datetime
import pandas 
import copy 

class Timer():

    def __init__(self):
        self.start_dt = None

    def start(self):
        self.start_dt = dt.datetime.now()

    def stop(self):
        end_dt = dt.datetime.now()
        print('Time taken: %s' % (end_dt - self.start_dt))
        
        

##################
# Name Generator ###############################################################
##################

def dir_name_creator(dir_name):
    counter = 0
    while os.path.isdir(dir_name):
        counter += 1
        dir_name = dir_name.split('(')[0]+'('+str(counter)+')'
    return dir_name

def dir_creator(dir_name,readme_text=None):
    dir_name = dir_name_creator(dir_name)
    os.mkdir(dir_name)
    if not readme_text==None:
        readme(dir_name+'/'+dir_name.split('/')[-1],readme_text)#
    print(dir_name,' Created!')
    return dir_name


def readme(readme_name,readme_text):
    f= open(readme_name+".txt","w+")
    f.write(readme_text)
    f.close()
    

def image_name_creator(image_name):
    '''
    Create the image name.
    '''
    counter = 0
    while os.path.isfile(image_name):
        counter += 1
        image_name, image_fromat = image_name.split('.')[0], image_name.split('.')[1]
        
        print(image_name,image_fromat)
        image_name = image_name.split('(')[0]+'('+str(counter)+').'+image_fromat
    print('Image name is :', image_name)
    return image_name


def save_image(image_name='image.jpg'):
    '''
    please use before the plt.show()
    save_image(image_name='image.jpg')
    '''
    image_name = image_name_creator(image_name)
    plt.savefig(image_name,bbox_inches = 'tight', pad_inches = 0.2, dpi=200)
    
    
    
    
def csv_file_name_creator(path, file_name, log=False):
    '''
    Create a file name.
    '''
    counter = 0
    while os.path.isfile(path+file_name):
        counter += 1
        file_name, file_fromat = file_name.split('.csv')[0], file_name.split('.csv')[1:]
        file_fromat = 'csv' + '.'.join(file_fromat)
        file_name = file_name.split('(')[0]+'('+str(counter)+').'+file_fromat
    print('File name is : '+str(file_name))    
    if log == True:
        logging.info('File name is : '+str( file_name))
    return file_name


def file_name_creator(path, file_name, log=False):
    '''
    Create a file name.
    '''
    counter = 0
    while os.path.isfile(path+file_name):
        counter += 1
        file_name, file_fromat = file_name.split('.')[0], file_name.split('.')[1]
        file_name = file_name.split('(')[0]+'('+str(counter)+').'+file_fromat
    print('File name is : '+str(file_name))    
    if log == True:
        logging.info('File name is : '+str(file_name))
    return file_name



##############
#   PANDAS   ######################################################################
##############


def update_col_name(lst_1, lst_2, suffixe):
    new_lst_1 = []
    for x in lst_1:
        if x in lst_2:
            x = str(x) + '_' + str(suffixe)
        new_lst_1.append(x)
    return new_lst_1



def mrg(on, how,  df, *dataframe, suffixe_lst=None):
    '''
    mrg(on, how,  df, *dataframe)
    '''
    
    assert isinstance(df, pd.DataFrame), 'The df is not pd.DataFrame!!! Please check first item.'
    for item, data in enumerate(dataframe):
        assert isinstance(data, pd.DataFrame), 'The data is not pd.DataFrame!!! Please check item '+str(item)
    

    if not on == None:
        right_index, left_index = False, False
    else:
        right_index, left_index = True, True
        

    
    for i, data in enumerate(dataframe):
        data = copy.copy(data)
        
        if suffixe_lst == None:
            data.columns = update_col_name(lst_1=data.columns, lst_2=df.columns, suffixe=str(i+1))
        else:
            data.columns = update_col_name(lst_1=data.columns, lst_2=df.columns, suffixe=suffixe_lst[i])
            
        
        df = pd.merge(df, data, right_index=right_index, left_index=left_index, how=how, on=on)

    return df


def _same_index_range(dt1, dt2):
    '''
    dt1, dt2 = same_index_range(dt1, dt2)
    '''
    
    dt1 = data_index_conv(dt1)
    dt2 = data_index_conv(dt2)
    
    dt1.sort_index(inplace=True)
    dt2.sort_index(inplace=True)
    
    if dt1.index[0] > dt2.index[0]:
        ind_strt = dt1.index[0]
    else:
        ind_strt = dt2.index[0]



    if dt1.index[-1] <  dt2.index[-1]:
        ind_end = dt1.index[-1]
    else:
        ind_end = dt2.index[-1]

    print(ind_strt,ind_end)

    dt1 = dt1.loc[ind_strt:ind_end,:] 
    dt2 = dt2.loc[ind_strt:ind_end,:] 
    
    return dt1, dt2
    
    
def same_index_range(df, *dataframs):
    
    temp = []
    result = []
    for dt in dataframs:
        df, dt = _same_index_range(df, dt)
        temp.append(dt)
    result.append(df)
    for dt in temp:
        df, dt = _same_index_range(df, dt)
        result[0] = df
        result.append(dt)
    
    return result


def data_index_conv(data):
    '''
    2020-03-10 10:00:00+01:00 ==> 2020-03-10 10:00:00
    2021-04-08 14:50:00+02:00 ==> 2021-04-08 14:50:00
    '''
    def index_conv(row):
        if isinstance(row.name, datetime.datetime) or isinstance(row.name, pandas.Timestamp):
            new_row_name = datetime.datetime.strftime(row.name, '%Y-%m-%d %H:%M:%S')
        elif isinstance(row.name, str):
#             new_row_name = datetime.datetime.strftime(datetime.datetime.strptime(row.name, '%Y-%m-%d %H:%M:%S'), '%Y-%m-%d %H:%M:%S')
            new_row_name = datetime.datetime.strftime(pd.to_datetime(row.name), '%Y-%m-%d %H:%M:%S')
        return new_row_name
    data.index = pd.to_datetime(data.apply(index_conv, axis=1))
    data.index.name = 'timestamp'
    return data






def corlag(data, period, step):
    '''
    Return Pearson product-moment correlation coefficients with different lag that period of lag defined in the "period" and 'step' defines the step that each lages have.
    corlag(data, period, step)
    data should be pd.DataFrame.
    '''

    def _crlg(data, col1, col2, period, step):
        
        index = 0
        res = pd.DataFrame(columns=['lag', col1+'_'+col2])
        
        for lag in range(0, period, step):
            
            
            if lag == 0 : 
                cc = np.corrcoef(data[col1].values, data[col2].values)[0][1]
            else: 
                cc = np.corrcoef(data[col1][:-lag].values, data[col2][lag:].values)[0][1]
                
                
            res.loc[index, ['lag',col1+'_'+col2]] = [lag, cc]
            index += 1 
        return res
    
    
    
    
    for col1 in data.columns: 
        for col2 in data.columns:
            if not 'rslts' in locals():
                rslts = _crlg(data=data, col1=col1, col2=col2, period=period, step=step)
            else:
                rslts = mrg('lag', 'outer', rslts, _crlg(data=data, col1=col1, col2=col2, period=period, step=step))
    return rslts


def node_selector(height=None, rack=None):
    pth_In_Temp = '/home/seyedkazemi/dataset/OY_BB_Inlet_Temp_dataset/roomM_BB_Inlet_Temp_1min_01_2019_to_01_2020_after_interpolate_sampeling_10T.csv'
    node_list = list(pd.read_csv(pth_In_Temp, index_col='index', nrows=1).columns.values)
    
    if not height==None:
        node_list = [node for node in node_list if 'c'+str(height) in node]
        
    if not rack==None:
        node_list = [node for node in node_list if 'r'+str(rack) in node]
        
    print(len(node_list),' nodes is selected.')
    print('For example:',random.sample(node_list, k = 4))
    assert  len(node_list)>=4, 'Please check the rack and height !!!!!!!!!!'
    
    return node_list



def df_shift_merge(dataframe, shift_range):
    merged_df = pd.DataFrame()
    
    for i in range(shift_range):
        merged_df = mrg(None, 'outer',  merged_df, dataframe.shift(i), suffixe_lst=['pst_'+str(i)]) 
    
    return merged_df



def check_same_columns(df1, df2):
    if len(df1.columns) != len(df2.columns):
        print('Size of columns should be same !!!')
    
    diff_col1 = [df1.columns[i] for i in range(len(df1.columns)) if df1.columns[i]!=df2.columns[i]]
    
    if len(diff_col1)!=0:
        print('Two dataframes should have same column names with same order !!!', diff_col1)
        

    
    diff_col2 = [df2.columns[i] for i in range(len(df2.columns)) if df1.columns[i]!=df2.columns[i]]
    
    if len(diff_col2)!=0:
        print('Two dataframes should have same column names with same order !!!', diff_col2)
        
    assert len(df1.columns) == len(df2.columns), 'Size of columns should be same !!!'
    assert len(diff_col1) == 0, 'Two dataframes should have same column names with same order !!!'+ str(diff_col1)
    assert len(diff_col2) == 0, 'Two dataframes should have same column names with same order !!!'+ str(diff_col2)
    
    
    
    
def diff_pd_dataframe(df1, df2):
    '''
    df = df1 - df2
    '''
    idx_1 = set(df1.index)
    idx_2 = set(df2.index)

    clmn_1 = set(df1.columns)
    clmn_2 = set(df2.columns)

    idx_1_dif_idx_2 = idx_1.difference(idx_2)
    idx_2_dif_idx_1 = idx_2.difference(idx_1)

    clmn_1_dif_clmn_2 = clmn_1.difference(clmn_2)
    clmn_2_dif_clmn_1 = clmn_2.difference(clmn_1)

    assert df1.shape == df2.shape, 'The shape of two dataframe should be same. df1.shape= ' + str(df1.shape) + 'df2.shape= ' + str(df2.shape)
    assert len(idx_1) == len(df1.index)  and len(idx_2) == len(df2.index), 'The index values should be unique.' 
    assert len(clmn_1) == len(df1.columns) and len(clmn_2) == len(df2.columns), 'The columns values should be unique.'
    assert len(idx_1_dif_idx_2) == 0, 'The index of two dataframe should be same. idx_1_dif_idx_2= ' + str(idx_1_dif_idx_2)
    assert len(idx_2_dif_idx_1) == 0, 'The index of two dataframe should be same. idx_2_dif_idx_1= ' + str(idx_2_dif_idx_1)
    assert len(clmn_1_dif_clmn_2) == 0, 'The columns of two dataframe should be same. clmn_1_dif_clmn_2= ' + str(clmn_1_dif_clmn_2)
    assert len(clmn_2_dif_clmn_1) == 0, 'The columns of two dataframe should be same. clmn_2_dif_clmn_1= ' + str(clmn_2_dif_clmn_1)
    df = df1 - df2
    return df


def sum_pd_dataframe(df1, df2):
    '''
    df = df1 + df2
    '''
    idx_1 = set(df1.index)
    idx_2 = set(df2.index)

    clmn_1 = set(df1.columns)
    clmn_2 = set(df2.columns)

    idx_1_dif_idx_2 = idx_1.difference(idx_2)
    idx_2_dif_idx_1 = idx_2.difference(idx_1)

    clmn_1_dif_clmn_2 = clmn_1.difference(clmn_2)
    clmn_2_dif_clmn_1 = clmn_2.difference(clmn_1)

    assert df1.shape == df2.shape, 'The shape of two dataframe should be same. df1.shape= ' + str(df1.shape) + 'df2.shape= ' + str(df2.shape)
    assert len(idx_1) == len(df1.index)  and len(idx_2) == len(df2.index), 'The index values should be unique.' 
    assert len(clmn_1) == len(df1.columns) and len(clmn_2) == len(df2.columns), 'The columns values should be unique.'
    assert len(idx_1_dif_idx_2) == 0, 'The index of two dataframe should be same. idx_1_dif_idx_2= ' + str(idx_1_dif_idx_2)
    assert len(idx_2_dif_idx_1) == 0, 'The index of two dataframe should be same. idx_2_dif_idx_1= ' + str(idx_2_dif_idx_1)
    assert len(clmn_1_dif_clmn_2) == 0, 'The columns of two dataframe should be same. clmn_1_dif_clmn_2= ' + str(clmn_1_dif_clmn_2)
    assert len(clmn_2_dif_clmn_1) == 0, 'The columns of two dataframe should be same. clmn_2_dif_clmn_1= ' + str(clmn_2_dif_clmn_1)
    df = df1 + df2
    return df



def dataframe_splitter(dataframe, Time_Window=datetime.timedelta(days=7) - datetime.timedelta(minutes=1), verbose=True):
    
    verboseprint = print if verbose else lambda *a, **k: None

    
    idx_1 = dataframe.index[0]
    idx_2 = dataframe.index[0] + Time_Window

    idx_lst = []
    splitted_dataframe_lst = []

    verboseprint(dataframe.index)

    while idx_2 <= dataframe.index[-1]: 

        idx_lst.append((idx_1, idx_2))
        splitted_dataframe_lst.append(dataframe.loc[idx_1:idx_2, :])

        idx_1 = idx_2
        idx_2 +=  Time_Window

    return splitted_dataframe_lst, idx_lst






def pd_row_to_columns_convertor(dataframe, window):
    """_summary_
    window = 2
    
    imput 
    	0	1
    0	2	4
    1	2	1
    2	3	2
    3	3	1
    4	1	0
    
    output
        0	1	2	3	4	5	6	7
    0	2.0	4.0	NaN	NaN	NaN	NaN	NaN	NaN
    1	2.0	4.0	2.0	1.0	NaN	NaN	NaN	NaN
    2	2.0	4.0	2.0	1.0	3.0	2.0	NaN	NaN
    3	2.0	4.0	2.0	1.0	3.0	2.0	3.0	1.0
    4	2.0	1.0	3.0	2.0	3.0	1.0	1.0	0.0

    Args:
        dataframe (_type_): _description_
        window (_type_): _description_
    """
    def fun(row):
        return pd.Series(np.concatenate(dataframe.loc[row.name-window:row.name, :].values))

    return dataframe.apply(fun, axis=1)

