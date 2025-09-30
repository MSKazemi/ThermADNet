def core_pivot():        
    import sys, os
    sys.path.append('/home/seyedkazemi/codes/mskhelper/')
    import pandas as pd, numpy as np
    import logging
    import datetime, pytz
    import m100_preprocessing_helper
    import imp 
    imp.reload(m100_preprocessing_helper)



    for metric in ['p1_core0_temp','p1_core10_temp','p1_core11_temp','p1_core12_temp','p1_core13_temp','p1_core14_temp','p1_core15_temp','p1_core16_temp',
                   'p1_core17_temp','p1_core18_temp','p1_core19_temp','p1_core1_temp','p1_core20_temp','p1_core21_temp','p1_core22_temp','p1_core23_temp',
                   'p1_core2_temp','p1_core3_temp','p1_core4_temp','p1_core5_temp','p1_core6_temp','p1_core7_temp','p1_core8_temp','p1_core9_temp',
                   'p0_core0_temp','p0_core10_temp','p0_core11_temp','p0_core12_temp','p0_core13_temp','p0_core14_temp','p0_core15_temp','p0_core16_temp',
                   'p0_core17_temp','p0_core18_temp','p0_core19_temp','p0_core1_temp','p0_core20_temp','p0_core21_temp','p0_core22_temp','p0_core23_temp',
                   'p0_core2_temp','p0_core3_temp','p0_core4_temp','p0_core5_temp','p0_core6_temp','p0_core7_temp','p0_core8_temp','p0_core9_temp',
                   'gpu0_core_temp','gpu0_mem_temp','gpu1_core_temp','gpu1_mem_temp','gpu3_core_temp','gpu3_mem_temp','gpu4_core_temp','gpu4_mem_temp']:

        print('__'+metric+'__'+metric+'__'+metric+'__'+metric+'__'+metric+'__'+metric+'__')
        logging.info(str('__'+metric+'__'+metric+'__'+metric+'__'+metric+'__'+metric+'__'+metric+'__'))
        log_file_name = 'log_raw_data_cook_'+datetime.datetime.strftime(datetime.datetime.now(pytz.timezone('Europe/Rome')), 
                                                                          '%Y-%m-%d %H:%M:%S').replace(':','-').replace(' ','_')+'.log'
        logging.basicConfig(filename=log_file_name,level=logging.INFO, format='%(levelname)s:%(asctime)s:%(lineno)d:%(message)s')


        path = '/home/seyedkazemi/dataset/Marconi100_backup/ipmi_pub/'+metric+'/'


        if not os.path.isdir(path+'/'+metric+'_done/'):
            os.makedirs(path+'/'+metric+'_done/')

        if not os.path.isdir(path+'/'+metric+'_pivot/'):
            os.makedirs(path+'/'+metric+'_pivot/')    

        files_list = sorted([file for file in os.listdir(path) if '.csv.gz' in file])
        # files_list = ['ambient_2020-03-14_00-00-00_to_2020-03-15_00-00-00.csv.gz']
        for file in files_list:
            print(file)
            logging.info(str(file))
            data = pd.read_csv(path+file, compression='gzip', index_col='timestamp')
            print('Data shape: ', data.shape)
            logging.info('Data shape: '+ str(data.shape))


            data = data.pipe(m100_preprocessing_helper.remove_Unnamed, log=True).\
            pipe(m100_preprocessing_helper.unique_val, log=True).\
            pipe(m100_preprocessing_helper.pvt,index='timestamp', freq='10T', columns='node', values='value', log=True)

            data.to_csv(path+'/'+metric+'_pivot/'+file.split('.')[0]+'.csv.gz', compression='gzip')
            os.replace(path+file, path+metric+'_done/'+file)
            print(50*'#')
            logging.info(str(50*'#'))        
        
        
        
        
def CRAC_pivot():        
    import sys, os
    sys.path.append('/home/seyedkazemi/codes/mskhelper/')
    import pandas as pd, numpy as np
    import logging
    import datetime, pytz
    import m100_preprocessing_helper
    import imp 
    imp.reload(m100_preprocessing_helper)


    pub = 'vertiv_pub'

    for metric in ['Supply_Air_Temperature','Fan_Speed','Actual_Return_Air_Temperature_Set_Point','Adjusted_Humidity',
                   'Compressor_Utilization','Dehumidifier_Utilization','Ext_Air_Sensor_A_Humidity','Ext_Air_Sensor_A_Temperature',
                   'Free_Cooling_Status','Free_Cooling_Valve_Open_Position','Hot_Water___Hot_Gas_Valve_Open_Position','Filter_Pressure_Drop',
                   'Free_Cooling_Fluid_Temperature','Ext_Air_Sensor_C_Humidity','Supply_Air_Temperature_Set_Point','Return_Air_Temperature',
                   'Return_Humidity','Humidifier_Utilization','Underflow_Static_Pressure','Humidity_Set_Point','Reheat_Utilization','Ext_Air_Sensor_C_Temperature']:


        print('__'+metric+'__'+metric+'__'+metric+'__'+metric+'__'+metric+'__'+metric+'__')
        logging.info(str('__'+metric+'__'+metric+'__'+metric+'__'+metric+'__'+metric+'__'+metric+'__'))
        log_file_name = 'CRAC_log_raw_data_cook_'+datetime.datetime.strftime(datetime.datetime.now(pytz.timezone('Europe/Rome')), 
                                                                          '%Y-%m-%d %H:%M:%S').replace(':','-').replace(' ','_')+'.log'
        logging.basicConfig(filename=log_file_name,level=logging.INFO, format='%(levelname)s:%(asctime)s:%(lineno)d:%(message)s')


        path = os.path.join('/home/seyedkazemi/dataset/Marconi100_backup/',pub,metric,'')

        if not os.path.isdir(path+'/'+metric+'_done/'):
            os.makedirs(path+'/'+metric+'_done/')

        if not os.path.isdir(path+'/'+metric+'_pivot/'):
            os.makedirs(path+'/'+metric+'_pivot/')    

        files_list = sorted([file for file in os.listdir(path) if '.csv.gz' in file])
        for file in files_list:
            print(file)
            logging.info(str(file))

            data = pd.read_csv(path+file, compression='gzip', index_col='timestamp')

            data.loc[data['value'] == 'Unavailable', 'value'] = np.nan
            data.loc[data['value'] == 'unknown', 'value'] = np.nan
            data['value'] = data['value'].astype(float)



            print('Data shape: ', data.shape)
            logging.info('Data shape: '+ str(data.shape))


            data = data.pipe(m100_preprocessing_helper.remove_Unnamed, log=True).\
            pipe(m100_preprocessing_helper.unique_val, log=True).\
            pipe(m100_preprocessing_helper.pvt, index='timestamp', freq='10T', columns='device', values='value', log=True)

            data.to_csv(path+'/'+metric+'_pivot/'+file.split('.')[0]+'.csv.gz', compression='gzip')
            os.replace(path+file, path+metric+'_done/'+file)
            print(50*'#')
            logging.info(str(50*'#'))
            
        
def modbus_pivot():
    import sys, os
    sys.path.append('/home/seyedkazemi/codes/mskhelper/')
    import pandas as pd, numpy as np
    import logging
    import datetime, pytz
    import m100_preprocessing_helper
    import imp 
    imp.reload(m100_preprocessing_helper)


    pub = 'logics_pub'

    for metric in ['Tot_chiller', 'Tot_qpompe', 'Tot_ict', 'Tot_cdz']:


        print('__'+metric+'__'+metric+'__'+metric+'__'+metric+'__'+metric+'__'+metric+'__')
        logging.info(str('__'+metric+'__'+metric+'__'+metric+'__'+metric+'__'+metric+'__'+metric+'__'))
        log_file_name = 'modbus_log_raw_data_cook_'+datetime.datetime.strftime(datetime.datetime.now(pytz.timezone('Europe/Rome')), 
                                                                          '%Y-%m-%d %H:%M:%S').replace(':','-').replace(' ','_')+'.log'
        logging.basicConfig(filename=log_file_name,level=logging.INFO, format='%(levelname)s:%(asctime)s:%(lineno)d:%(message)s')


        path = os.path.join('/home/seyedkazemi/dataset/Marconi100_backup/',pub,metric,'')

        if not os.path.isdir(path+'/'+metric+'_done/'):
            os.makedirs(path+'/'+metric+'_done/')

        if not os.path.isdir(path+'/'+metric+'_pivot/'):
            os.makedirs(path+'/'+metric+'_pivot/')    

        files_list = sorted([file for file in os.listdir(path) if '.csv.gz' in file])
        for file in files_list:
            print(file)
            logging.info(str(file))

            data = pd.read_csv(path+file, compression='gzip', index_col='timestamp')

            data.loc[data['value'] == 'Unavailable', 'value'] = np.nan
            data.loc[data['value'] == 'unknown', 'value'] = np.nan
            data['value'] = data['value'].astype(float)



            print('Data shape: ', data.shape)
            logging.info('Data shape: '+ str(data.shape))


            data = data.pipe(m100_preprocessing_helper.remove_Unnamed, log=True).\
            pipe(m100_preprocessing_helper.unique_val, log=True).\
            pipe(m100_preprocessing_helper.pvt, index='timestamp', freq='10T', columns='name', values='value', log=True)

            data.to_csv(path+'/'+metric+'_pivot/'+file.split('.')[0]+'.csv.gz', compression='gzip')
            os.replace(path+file, path+metric+'_done/'+file)
            print(50*'#')
            logging.info(str(50*'#'))        

            
            
            
            
            
            
def raw_data_pivot():            
    import sys, os
    sys.path.append('/home/seyedkazemi/codes/mskhelper/')
    import pandas as pd, numpy as np
    import logging
    import datetime, pytz
    import m100_preprocessing_helper
    import imp 
    imp.reload(m100_preprocessing_helper)

    for metric in ['ambient',
                   'fan0_0','fan0_1','fan1_0','fan1_1','fan2_0','fan2_1','fan3_0','fan3_1',
                   'pcie',
                   'ps0_input_power','ps1_input_power']:

    # for metric in ['ambient',
    #                'fan0_0','fan0_1','fan1_0','fan1_1','fan2_0','fan2_1','fan3_0','fan3_1',
    #                'pcie',
    #                'ps0_input_power','ps1_input_power',
    #                'gpu0_core_temp','gpu0_mem_temp','gpu1_core_temp','gpu1_mem_temp','gpu3_core_temp','gpu3_mem_temp','gpu4_core_temp','gpu4_mem_temp',
    #                'p0_core0_temp','p0_core10_temp','p0_core11_temp','p0_core12_temp','p0_core13_temp','p0_core14_temp','p0_core15_temp','p0_core16_temp',
    #                'p0_core17_temp','p0_core18_temp','p0_core19_temp','p0_core1_temp','p0_core20_temp','p0_core21_temp','p0_core22_temp','p0_core23_temp',
    #                'p0_core2_temp','p0_core3_temp','p0_core4_temp','p0_core5_temp','p0_core6_temp','p0_core7_temp','p0_core8_temp','p0_core9_temp',
    #                'p0_vdd_temp',
    #                'p1_core0_temp','p1_core10_temp','p1_core11_temp','p1_core12_temp','p1_core13_temp','p1_core14_temp','p1_core15_temp','p1_core16_temp',
    #                'p1_core17_temp','p1_core18_temp','p1_core19_temp','p1_core1_temp','p1_core20_temp','p1_core21_temp','p1_core22_temp','p1_core23_temp',
    #                'p1_core2_temp','p1_core3_temp','p1_core4_temp','p1_core5_temp','p1_core6_temp','p1_core7_temp','p1_core8_temp','p1_core9_temp',
    #                'p1_vdd_temp']:

        print('__'+metric+'__'+metric+'__'+metric+'__'+metric+'__'+metric+'__'+metric+'__')
        logging.info(str('__'+metric+'__'+metric+'__'+metric+'__'+metric+'__'+metric+'__'+metric+'__'))
        log_file_name = 'log_raw_data_cook_'+datetime.datetime.strftime(datetime.datetime.now(pytz.timezone('Europe/Rome')), 
                                                                          '%Y-%m-%d %H:%M:%S').replace(':','-').replace(' ','_')+'.log'
        logging.basicConfig(filename=log_file_name,level=logging.INFO, format='%(levelname)s:%(asctime)s:%(lineno)d:%(message)s')


        path = '/home/seyedkazemi/dataset/Marconi100_backup/ipmi_pub/'+metric+'/'


        if not os.path.isdir(path+'/'+metric+'_done/'):
            os.makedirs(path+'/'+metric+'_done/')

        if not os.path.isdir(path+'/'+metric+'_pivot/'):
            os.makedirs(path+'/'+metric+'_pivot/')    

        files_list = sorted([file for file in os.listdir(path) if '.csv.gz' in file])
        # files_list = ['ambient_2020-03-14_00-00-00_to_2020-03-15_00-00-00.csv.gz']
        for file in files_list:
            print(file)
            logging.info(str(file))
            data = pd.read_csv(path+file, compression='gzip', index_col='timestamp')
            print('Data shape: ', data.shape)
            logging.info('Data shape: '+ str(data.shape))


            data = data.pipe(m100_preprocessing_helper.remove_Unnamed, log=True).\
            pipe(m100_preprocessing_helper.unique_val, log=True).\
            pipe(m100_preprocessing_helper.pvt,index='timestamp', freq='10T', columns='node', values='value', log=True)

            data.to_csv(path+'/'+metric+'_pivot/'+file.split('.')[0]+'.csv.gz', compression='gzip')
            os.replace(path+file, path+metric+'_done/'+file)
            print(50*'#')
            logging.info(str(50*'#'))


            
            
            
def RDHX_pivot():            
    import sys, os
    sys.path.append('/home/seyedkazemi/codes/mskhelper/')
    import pandas as pd, numpy as np
    import logging
    import datetime, pytz
    import m100_preprocessing_helper
    import imp 
    imp.reload(m100_preprocessing_helper)


    pub = 'schneider_pub'

    for metric in ['PLC_PLC_Q101.Portata_1_hmi','PLC_PLC_Q101.Portata_2_hmi',
                   'PLC_PLC_Q101.Temp_mandata','PLC_PLC_Q101.Temp_ritorno',
                   'PLC_PLC_Q101.Posizione_ty141','PLC_PLC_Q101.Posizione_ty142',
                   'PLC_PLC_Q101.T_ritorno_hmi','PLC_PLC_Q101.T_mandata_hmi',
                   'PLC_PLC_Q101.Portata_1','PLC_PLC_Q101.Portata_2',
                   'PLC_PLC_Q101.Delta_temp']:


        print('__'+metric+'__'+metric+'__'+metric+'__'+metric+'__'+metric+'__'+metric+'__')
        logging.info(str('__'+metric+'__'+metric+'__'+metric+'__'+metric+'__'+metric+'__'+metric+'__'))
        log_file_name = 'CRAC_log_raw_data_cook_'+datetime.datetime.strftime(datetime.datetime.now(pytz.timezone('Europe/Rome')), 
                                                                          '%Y-%m-%d %H:%M:%S').replace(':','-').replace(' ','_')+'.log'
        logging.basicConfig(filename=log_file_name,level=logging.INFO, format='%(levelname)s:%(asctime)s:%(lineno)d:%(message)s')


        path = os.path.join('/home/seyedkazemi/dataset/Marconi100_backup/',pub,metric,'')

        if not os.path.isdir(path+'/'+metric+'_done/'):
            os.makedirs(path+'/'+metric+'_done/')

        if not os.path.isdir(path+'/'+metric+'_pivot/'):
            os.makedirs(path+'/'+metric+'_pivot/')    

        files_list = sorted([file for file in os.listdir(path) if '.csv.gz' in file])
        for file in files_list:
            print(file)
            logging.info(str(file))

            data = pd.read_csv(path+file, compression='gzip', index_col='timestamp')

            data.loc[data['value'] == 'Unavailable', 'value'] = np.nan
            data.loc[data['value'] == 'unknown', 'value'] = np.nan
            data['value'] = data['value'].astype(float)



            print('Data shape: ', data.shape)
            logging.info('Data shape: '+ str(data.shape))


            data = data.pipe(m100_preprocessing_helper.remove_Unnamed, log=True).\
            pipe(m100_preprocessing_helper.unique_val, log=True).\
            pipe(m100_preprocessing_helper.pvt, index='timestamp', freq='10T', columns='panel', values='value', log=True)

            data.to_csv(path+'/'+metric+'_pivot/'+file.split('.')[0]+'.csv.gz', compression='gzip')
            os.replace(path+file, path+metric+'_done/'+file)
            print(50*'#')
            logging.info(str(50*'#'))
            
            
            
            
def weather_pivot():
    import sys, os
    sys.path.append('/home/seyedkazemi/codes/mskhelper/')
    import pandas as pd, numpy as np
    import logging
    import datetime, pytz
    import m100_preprocessing_helper
    import imp 
    imp.reload(m100_preprocessing_helper)


    pub = 'weather_pub'

    for metric in ['temp']:


        print('__'+metric+'__'+metric+'__'+metric+'__'+metric+'__'+metric+'__'+metric+'__')
        logging.info(str('__'+metric+'__'+metric+'__'+metric+'__'+metric+'__'+metric+'__'+metric+'__'))
        log_file_name = 'weather_log_raw_data_cook_'+datetime.datetime.strftime(datetime.datetime.now(pytz.timezone('Europe/Rome')), 
                                                                          '%Y-%m-%d %H:%M:%S').replace(':','-').replace(' ','_')+'.log'
        logging.basicConfig(filename=log_file_name,level=logging.INFO, format='%(levelname)s:%(asctime)s:%(lineno)d:%(message)s')


        path = os.path.join('/home/seyedkazemi/dataset/Marconi100_backup/',pub,metric,'')

        if not os.path.isdir(path+'/'+metric+'_done/'):
            os.makedirs(path+'/'+metric+'_done/')

        if not os.path.isdir(path+'/'+metric+'_pivot/'):
            os.makedirs(path+'/'+metric+'_pivot/')    

        files_list = sorted([file for file in os.listdir(path) if '.csv.gz' in file])
        for file in files_list:
            print(file)
            logging.info(str(file))

            data = pd.read_csv(path+file, compression='gzip', index_col='timestamp')

            data.loc[data['value'] == 'Unavailable', 'value'] = np.nan
            data.loc[data['value'] == 'unknown', 'value'] = np.nan
            data['value'] = data['value'].astype(float)



            print('Data shape: ', data.shape)
            logging.info('Data shape: '+ str(data.shape))


            data = data.pipe(m100_preprocessing_helper.remove_Unnamed, log=True).\
            pipe(m100_preprocessing_helper.unique_val, log=True).\
            pipe(m100_preprocessing_helper.pvt, index='timestamp', freq='10T', columns='type', values='value', log=True)

            data.to_csv(path+'/'+metric+'_pivot/'+file.split('.')[0]+'.csv.gz', compression='gzip')
            os.replace(path+file, path+metric+'_done/'+file)
            print(50*'#')
            logging.info(str(50*'#'))

            
            
            
import datetime, pytz, time        
now = datetime.datetime.now(pytz.timezone('Europe/Rome'))

while True:
    print(now)
    if now.hour == 16: 
        CRAC_pivot()
        modbus_pivot()
        raw_data_pivot()
        RDHX_pivot()
        weather_pivot()
        core_pivot()
    time.sleep(60*60)    
    now = datetime.datetime.now(pytz.timezone('Europe/Rome'))        