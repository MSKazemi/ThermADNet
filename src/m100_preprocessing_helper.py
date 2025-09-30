import os,sys,glob
import pandas as pd
import numpy as np
import os
# import dask
# from dask import dataframe as dsf
sys.path.append('/home/seyedkazemi/codes/HPCRoomModel/mskhelper/')
import mohsenutils
import logging


def remove_Unnamed(df, log=False):
    if 'Unnamed: 0' in df.columns:
        df.drop('Unnamed: 0', axis=1, inplace=True)
        if log == True:
            logging.info("'Unnamed: 0' ==> Dropped !!!")
    return df



def unique_val(df, log=False):
    columns = list(df.columns)
    print(columns)
    if log == True:
        logging.info(columns)
    if 'value' in columns:
        columns.remove('value')
    if 'timestamp' in columns:
        columns.remove('timestamp')
    print(columns)
    if log == True:
        logging.info(columns)
    for col in columns:
        if col == 'node':
            print(col,'==>',len(df[col].unique()))
            if log == True:
                logging.info(str(col)+'==>'+str(len(df[col].unique())))
        else:
            print(col,'==>',df[col].unique())
            if log == True:
                logging.info(str(col)+'==>'+str(df[col].unique()))
    return df



# def to_category(df):
#     cols = list(df.columns)
#     for col in cols:
#         ratio = len(df[col].value_counts()) / len(df)
#         if ratio < 0.05:
#             df[col] = df[col].astype('category')
#     print(data.info())
#     logging.info(str(data.info()))
#     return df


    
    


def pvt(df, freq, index=None, columns='node', values='value', log=False):
    df = df.pivot_table(index=index,columns=columns, values=values)
    print(df)
    print('Data shape after pivot: ', df.shape)
    df.index = pd.to_datetime(df.index)
    df = mohsenutils.data_index_conv(df)
    if log == True:
        logging.info('Data shape after pivot: '+str(df.shape))
    df = df.resample(freq).mean()
    print('Data shape after resample: ', df.shape)
    if log == True:
        logging.info('Data shape after resample: '+str(df.shape))
    df.index.name = 'timestamp'
    return df



def concat_multiple_csvs(csvdir, big_csv_path_name, old_pivot_path, freq):
    
#     assert not os.path.isfile(str(big_csv_path_name)), 'There is file with the same name in the ' + str(big_csv_path_name)
    
    csvfiles = glob.glob(os.path.join(csvdir, '*.csv.gz'))
    print(csvfiles)
    dataframes = []  
    for csvfile in csvfiles:
        df = pd.read_csv(csvfile)
        dataframes.append(df)
    print('Number of file', len(dataframes))
    if len(dataframes) > 1:
        result = pd.concat(dataframes)
        result['timestamp'] = pd.to_datetime(result['timestamp'])
        result.set_index('timestamp', inplace=True)
        result.sort_index(inplace=True)
        result = result.resample(freq).mean()
        result.sort_index(inplace=True)
        result.to_csv(big_csv_path_name, compression='gzip')
        print(str(big_csv_path_name), 'Created !!!')

#         old_pivot_path = csvdir.split('_pivot/')[0]+'_old_pivot/'


        if not os.path.isdir(old_pivot_path):
            os.makedirs(old_pivot_path)

        for csvfile in csvfiles:
            if not 'big' in csvfile:
                os.replace(csvfile, old_pivot_path+csvfile.split('/')[-1])
    else:
        print('Your pivot already has been updated.')
        result = ''
    
    return result


# def update_index_name(data):
#     '''
#     '''
#     if 'Unnamed: 0' in list(data.columns):
#         data.rename(columns={'Unnamed: 0':'timestamp'}, inplace=True)
#         data.set_index('timestamp', inplace=True)
#     elif 'timestamp' in list(data.columns):
#         data.set_index('timestamp', inplace=True)
#     elif 'index' in list(data.columns):
#         data.rename(columns={'index':'timestamp'}, inplace=True)
#         data.set_index('timestamp', inplace=True)
#     else:
#         print('The column with name "Unnamed: 0" or "timestamp" or "index" is not inside the columns od data')
#         print(data.columns)
#         print(50*'++')
#         sys.exit()
#     return data
        

# def update_index_name_all_files(path):
#     files_list = sorted([file for file in os.listdir(path) if ('.csv' in file) and not ('.csv.gz' in file)])
#     for file in files_list:
#         print(file)
#         data = pd.read_csv(path+file)
#         data = update_index_name(data)
#         data.to_csv(path+file)


# def create_dataframe_template_with_all_columns(path, metric_name, index_name):
    
#     '''
#     results = create_dataframe_template_with_all_columns(path, metric_name, index_name='timestamp')
#     '''
#     assert not os.path.isfile(path+str(metric_name)+'_template.csv'), 'There is file with the same name in the '+str(path+str(metric_name)+'_template.csv')
#     lf_lst = sorted([file for file in os.listdir(path) if ('.csv' in file) and (str(metric_name) in file) and not ('.csv.gz' in file)])
#     print(len(lf_lst))
#     clmns = []
#     max_col = 0
#     for i, f in enumerate(lf_lst):
#         print(f)
#         col = pd.read_csv(path+f, nrows=1, index_col=index_name).columns.values
#         if max_col < len(col):
#             max_col = len(col)
#         print(i, len(col), max_col)
#         clmns.append(col)

#     set_clmns = set(np.concatenate(clmns))
#     print(len(np.concatenate(clmns)), len(set_clmns))
#     results = pd.DataFrame(columns=set_clmns)
#     results.index.name = index_name
#     if not os.path.isfile(path+str(metric_name)+'_template.csv'):
#         results.to_csv(path+str(metric_name)+'_template.csv')
#     else:
#         print('There is file with the same name in the ',path+str(metric_name)+'_template.csv')
#     return results



# def update_columns(original_data_path, new_data_path, template_name, index_name='timestamp'):
    
#     '''
#     update_columns(original_data_path, new_data_path, template_name, index_name='timestamp')
#     '''
#     assert not os.path.isdir(str(new_data_path)), 'There is a directory with the same name in the '+str(new_data_path)

#     lf_lst = [fl for fl in os.listdir(original_data_path) if '.csv' in fl]
#     print(len(lf_lst))
#     lf_lst.remove(str(template_name))
#     print(len(lf_lst))
    
#     tmplt = pd.read_csv(original_data_path+str(template_name), index_col=str(index_name))
#     os.makedirs(str(new_data_path))
#     for i, f in enumerate(lf_lst):
#         print(i, f)
#         data = pd.read_csv(str(original_data_path)+str(f), index_col=str(index_name))
#         dt = pd.concat((tmplt, data))
#         dt.to_csv(str(new_data_path)+str(f))
#         print(data.shape, dt.shape)
        

# This function can have error due to the fact the dask did not consider the header of the second and ... csv file and if the number of the columns will be same it does not check the name of the columns ... so be aware about using this function if you are sure that the headers are same you can use this function.

# def create_one_big_file(path, new_big_file_path_name):

#     '''
#     data = create_one_big_file(path, new_big_file_path_name)
#     '''
#     assert not os.path.isfile(str(new_big_file_path_name)), 'There is file with the same name in the ' + str(new_big_file_path_name)

#     data = dsf.read_csv(path+'*.csv')
#     display(data)
#     data = data.compute()
#     display(data)
#     data.set_index('timestamp', inplace=True)
#     data.sort_index(inplace=True)
#     display('After sort:', data)
#     print(data.shape)
# #     data.index = pd.to_datetime(data.index)
#     data = mohsenutils.data_index_conv(data)
    
    
#     data.sort_index(inplace=True)
#     data = data.resample('10T').mean()
#     print('Resample 10T',data.shape)
#     display('Resample 10T', data)
#     data.sort_index(inplace=True)
#     data.to_csv(str(new_big_file_path_name))
#     return data





###########
#  FLAGS  #
###########


def flag_min_chassis_of_rack(data):
    result = pd.DataFrame()
    racks = list(set([c.split('n')[0].split('r')[1] for c in [c for c in data.columns if 'r' in c]]))
    print(racks)
    def _min_chassis(row):
        try:
            min_chassis =  int(row.split('n')[1])
        except:
            min_chassis = np.nan
        return min_chassis

    for rck in racks:
        chassis_of_rck = [chassis for chassis in data.columns if rck in chassis]
        df = pd.DataFrame(data[chassis_of_rck].idxmin(axis=1), columns=[str(rck)])
        df[rck] = df[rck].apply(_min_chassis)
        result = mohsenutils.mrg(None, 'outer', df, result)
        print(rck, len(chassis_of_rck),df.shape, result.shape)
    return result




def flag_pd_data_horiz_rank(data):
    '''
    receive a Pnadas Dataframe then return the an other Pandas DataFrame which the number of the celles show the horizontal ranck of the cell based on the numeric value.
    for example, [15 1 80 3] ==> [2 0 3 1]
    '''
    
    def _rank(row):
#         global e
        sorted_row = sorted(row)
#         if len(list(set(sorted_row))) != len(row):
#             print(e,'Error, There are two or more columns with same Number !!! ')
#             e += 1
#         assert len(list(set(sorted_row))) == len(row), 'Error, There are two or more columns with same Number !!! '

        try:
            rank_row = [sorted_row.index(value) for value in row]
            for i, index in enumerate(row.index):

                result.loc[row.name,index] = int(rank_row[i])
        except:
            for i, index in enumerate(row.index):
                result.loc[row.name,index] = np.nan
        

    result = pd.DataFrame(columns=data.columns)
    data.apply(_rank, axis=1)
    result.index.name = 'timestamp'
    return result

