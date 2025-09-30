import os, sys, imp, glob
try:
    # Get the absolute path to the 'libs' directory based on the script location
    libs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../mskhelper'))
except NameError:
    # Fallback for interactive mode
    libs_dir = os.path.abspath(os.path.join(os.getcwd(), '../../mskhelper'))

print(libs_dir)
all_items = os.listdir(libs_dir)
print(all_items)
sys.path.append(libs_dir)
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../DNN')))
sys.path.append('/home/seyedkazemi/codes/mskhelper/')
sys.path.append('/home/seyedkazemi/codes/Marconi100/DNN/')
sys.path.append('/home/seyedkazemi/codes/Marconi100/')
import pandas as pd, numpy as np
import datetime, pytz
import data_provider, mohsenutils, m100_preprocessing_helper, flag_helper, autoencoder_helper, function_set
import imp 
import datetime
from  matplotlib import pyplot as plt
import seaborn as sns
imp.reload(data_provider)
imp.reload(mohsenutils)
imp.reload(m100_preprocessing_helper)
imp.reload(flag_helper)
imp.reload(autoencoder_helper)
imp.reload(function_set)


kwargs = {'parse_dates':True, 'date_parser':pd.to_datetime, 'index_col':'timestamp'}
p1 = {'model_dir_path':'/scratch/seyedkazemi/Marconi100-Cooked/AE/LSTM-AE-5-Month/LSTM-AE_Results_SqL2-ftur242grey_data_range_01_25-f10/Train_2021-06-15_to_2021-07-15/', 
      'label':'Training Period: 2021-06-15 to 2021-07-15',
      'start_train':'2021-06-15 00:00:00',
      'stop_train' :'2021-07-15 00:00:00',
      'color':'green',
      'experiment':'Experiment 1'}
p2 = {'model_dir_path' : '/scratch/seyedkazemi/Marconi100-Cooked/AE/LSTM-AE-5-Month/LSTM-AE_Results_SqL2-ftur242grey_data_range_01_25-f10/Train_2021-04-08_to_2021-07-01/', 
      'label' : 'Training Period: 2021-04-08 to 2021-07-01',
      'start_train':'2021-04-08 00:00:00',
      'stop_train' :'2021-07-01 00:00:00',
      'color':'brown',
      'experiment':'Experiment 2'}
p3 = {'model_dir_path' : '/scratch/seyedkazemi/Marconi100-Cooked/AE/LSTM-AE-5-Month/LSTM-AE_Results_SqL2-ftur242grey_data_range_01_25-f10/Train_2021-04-08_to_2021-07-15/', 
      'label' : 'Training Period: 2021-04-08 to 2021-07-15',
      'start_train':'2021-04-08 00:00:00',
      'stop_train' :'2021-07-15 00:00:00',
      'color':'yellow', 
      'experiment':'Experiment 3'}

p4 = {'model_dir_path' : '/scratch/seyedkazemi/Marconi100-Cooked/AE/LSTM-AE-5-Month/LSTM-AE_Results_SqL2-ftur242grey_data_range_01_25-f10/Train_2021-04-08_to_2021-05-22/', 
      'label' : 'Training Period: 2021-04-08 to 2021-05-22',
      'start_train':'2021-04-08 00:00:00',
      'stop_train' :'2021-05-22 00:00:00',
      'color':'cyan',
      'experiment':'Experiment 4'}
p5 = {'model_dir_path' : '/scratch/seyedkazemi/Marconi100-Cooked/AE/LSTM-AE-5-Month/LSTM-AE_Results_SqL2-ftur242grey_data_range_01_25-f10/Train_2021-04-08_to_2021-07-22/', 
      'label' : 'Training Period: 2021-04-08 to 2021-07-22',
      'start_train':'2021-04-08 00:00:00',
      'stop_train' :'2021-07-22 00:00:00',
      'color':'darkorange',
      'experiment':'Experiment 5'}
p6 = {'model_dir_path' : '/scratch/seyedkazemi/Marconi100-Cooked/AE/LSTM-AE-5-Month/LSTM-AE_Results_SqL2-ftur242grey_data_range_01_25-f10/Train_2021-06-22_to_2021-07-22/', 
      'label' : 'Training Period: 2021-06-22 to 2021-07-22',
      'start_train':'2021-06-22 00:00:00',
      'stop_train' :'2021-07-22 00:00:00',
      'color':'blue',
      'experiment':'Experiment 6'}
p7 = {'model_dir_path' : '/scratch/seyedkazemi/Marconi100-Cooked/AE/LSTM-AE-5-Month/LSTM-AE_Results_SqL2-ftur242grey_data_range_01_25-f10/Train_2021-04-08_to_2021-07-27/', 
      'label' : 'Training Period: 2021-04-08 to 2021-07-27',
      'start_train':'2021-04-08 00:00:00',
      'stop_train' :'2021-07-27 00:00:00',
      'color':'olive',
      'experiment':'Experiment 7'}
p8 = {'model_dir_path':'/scratch/seyedkazemi/Marconi100-Cooked/AE/LSTM-AE-4_Train_2021-04-08-23-30_to_2021-08-01-23-50/LSTM-AE_Results_SqL2-ftur242grey_data_range_01_25-f10/', 
      'label':'Training Period: 2021-04-08 to 2021-08-02', 
      'start_train':'2021-04-08 00:00:00', 
      'stop_train' :'2021-08-02 00:00:00',
      'color':'red',
      'experiment':'Experiment 8'}
p9 = {'model_dir_path' : '/scratch/seyedkazemi/Marconi100-Cooked/AE/LSTM-AE-5-Month/LSTM-AE_Results_SqL2-ftur242grey_data_range_01_25-f10/Train_2021-04-08_to_2021-08-18/', 
      'label' : 'Training Period: 2021-04-08 to 2021-08-18',
      'start_train':'2021-04-08 00:00:00',
      'stop_train' :'2021-08-18 00:00:00',
      'color':'blueviolet', 
      'experiment':'Experiment 9'}





data_path_lst = [p1, p2, p3, p4, p5, p6, p7, p8, p9]

dataframe_lst = [pd.read_csv(dt_pth['model_dir_path']+'/Error_Location/Inference_Inference_to_2021_08_17_23_50.csv', **kwargs) for dt_pth in data_path_lst]


grey_data_range_train_threshold=[1, 25, 10]




for i in range(len(data_path_lst)):
    autoencoder_helper.add_label(df=dataframe_lst[i], 
                                 grey_data_range_train_threshold=grey_data_range_train_threshold, 
                                 verbose=False, 
                                 start_train=data_path_lst[i]['start_train'], 
                                 stop_train=data_path_lst[i]['stop_train'])
    dataframe_lst[i]['class_flags'] = dataframe_lst[i][['sum_flags']].apply(autoencoder_helper.classifier_flag, 
                                                                            args=grey_data_range_train_threshold, 
                                                                            axis=1) 

print(dataframe_lst[0].shape)
indx = dataframe_lst[0].index
print(len(indx))
    
################
#  Parameters  #
################     
    
zoom = True
qntl = 0.99
legend_fontsize = 20
fontsize_legend_1 = 17
ylabel_fontsize = 20
alpha=0.7
lw=3

fig_Title = "Error Threshold Quantile("+str(qntl)+") of $10<\sum{}Flags<25$"
if zoom:
    fig_name = '9Exp_10SumFlags25_zoom'
else:
    fig_name = '9Exp_10SumFlags25'
error_threshold_lst = [dataframe_lst[i].query('Label=="Train_Month_grey"')[['Error']].quantile(qntl) for i in range(len(data_path_lst))]



# fig_Title = "Error Threshold Quantile("+str(qntl)+") of $0\leq\sum{}Flags\leq10$"
# if zoom:
#     fig_name = '9Exp_0SumFlags10_zoom'
# else:
#     fig_name = '9Exp_0SumFlags10'
# error_threshold_lst = [dataframe_lst[i].query('Label=="Train_Month_train" or Label=="Train_Month_grey_train"')[['Error']].quantile(qntl) for i in range(len(data_path_lst))]






# fig_Title = "Error Threshold Quantile("+str(qntl)+") of $0<\sum{}Flags<25$"
# if zoom:
#     fig_name = 'fig2_zoom.jpg'
# else:
#     fig_name = 'fig2.jpg'
# error_threshold_lst = [dataframe_lst[i].query('Label=="Train_Month_grey" or Label=="Train_Month_grey_train"')[['Error']].quantile(qntl) for i in range(len(data_path_lst))]


# fig_Title ="Error Threshold Quantile("+str(qntl)+") of $\sum{}Flags=0$"
# if zoom:
#     fig_name = 'fig4_zoom.jpg'
# else:
#     fig_name = 'fig4.jpg'
# error_threshold_lst = [dataframe_lst[i].query('Label=="Train_Month_train"')[['Error']].quantile(qntl) for i in range(len(data_path_lst))]






for i in range(len(data_path_lst)):    
    dataframe_lst[i]['class_error'] = dataframe_lst[i][['Error']].apply(autoencoder_helper.classifier_error, args=error_threshold_lst[i].values, axis=1) 
    

##########################
#  Drop the Missed Data  #
##########################
# index_list_null_removed = pd.read_csv('index_list_null_removed.csv', index_col='index_list_null_removed', parse_dates=True, date_parser=pd.to_datetime).index
# null_index = pd.read_csv('null_index.csv', index_col='null_index', parse_dates=True, date_parser=pd.to_datetime).index

# for i in range(len(data_path_lst)):
#     dataframe_lst[i].drop(index=null_index, inplace=True)
#     dataframe_lst[i] = mohsenutils.mrg(None, 'outer', *[pd.DataFrame(index=indx), dataframe_lst[i]])
##########################
#  Drop the Missed Data  #
##########################
    

start = pd.to_datetime("2021-07-25 00:00:00")
stop  = pd.to_datetime("2021-07-30 00:00:00")

# display('1Month',Inf_Trn_06_22_to_07_22.mean(),
#         '4Month',Inf_Trn_04_08_to_08_01.mean(), 
#         '3Month',Inf_Trn_04_08_to_07_15.mean())

# print('Mean class_flags',len(Inf_Trn_06_22_to_07_22[Inf_Trn_06_22_to_07_22['class_flags']==1])/(len(Inf_Trn_06_22_to_07_22)))

rows = 12
s_lbl = 0.2
height_ratios = [0.4,0.7,s_lbl,s_lbl,s_lbl,s_lbl,s_lbl,s_lbl,s_lbl,s_lbl,s_lbl,s_lbl]
# height_ratios = [1 for i in range(rows)]

fig, axl = plt.subplots(rows,1,figsize=(25, rows*3), sharex=True, gridspec_kw={'height_ratios':height_ratios})

# fig.suptitle(fig_Title, fontsize=35)


ll0_0 = axl[0].plot(dataframe_lst[0].sum_flags, color='green', label='$\sum_{}^{} Flags $',lw=lw)
axl[0].set_ylabel('$\sum_{}^{} Flags $', fontsize=ylabel_fontsize)
axl[0].legend(fontsize=legend_fontsize,loc='upper left')


# axr = [axl[0].twinx()]
# lr0_0 = axr[0].plot(Inf_Trn_06_22_to_07_22['sum_flags'].rolling(window='7D').mean(),color='orange', lw=4, label='Moving Average of $\sum_{}^{} Flags $\nWindow = One Week')
# axr[0].set_ylabel('Moving Average of $\sum_{}^{} Flags $\nWindow = One Week', fontsize=22)
# axr[0].tick_params(axis='y', labelsize=20, colors='orange') 
# axr[0].spines['right'].set_color('orange')

# lns0 = ll0_0 + lr0_0
# labs0 = [l.get_label() for l in lns0]
# axl[0].legend(lns0, labs0, loc=0, fontsize=17)

lns1 = []

for i in range(len(data_path_lst)):
    ll1 = axl[1].plot(dataframe_lst[i].Error, color=str(data_path_lst[i]['color']), label='LSTM-AE - '+str(data_path_lst[i]['label']), alpha=alpha,lw=lw)
    lns1 += ll1
#     axl[1].axhline(error_threshold_lst[i].values, color=str(data_path_lst[i]['color']))
    print(data_path_lst[i]['label'],error_threshold_lst[i].values)
    
labs1 = [l.get_label() for l in lns1]
legend1 = axl[1].legend(lns1, labs1, loc='upper left', fontsize=fontsize_legend_1)
legend1.get_frame().set_alpha(0.3)

axl[1].set_yscale('log')
axl[1].set_ylabel('Reconstruction Error $\\varepsilon$ ', fontsize=ylabel_fontsize)



axl[2].plot(dataframe_lst[0].class_flags, color='green',label='Label generated by: $\sum_{}^{} Flags $',lw=lw) 
y_ticks = [0, 0.5, 1]
y_labels = ['Normal','Ambiguous', 'Abnormal']
axl[2].set_yticks(y_ticks)
axl[2].set_yticklabels(y_labels)
axl[2].legend(fontsize=legend_fontsize)

for i in range(len(data_path_lst)):
    axl[3+i].plot(dataframe_lst[i].class_error, color=data_path_lst[i]['color'], label='Label generated by: LSTM-AE '+str(data_path_lst[i]['label']), lw=lw)
    axl[3+i].axvline(pd.to_datetime(data_path_lst[i]['start_train']), color='k', lw=6, linestyle='dashed')  #str(data_path_lst[i]['color'])
    axl[3+i].axvline(pd.to_datetime(data_path_lst[i]['stop_train']),  color='k', lw=6, linestyle='dashed')  #str(data_path_lst[i]['color'])
    axl[3+i].axvline(pd.to_datetime(data_path_lst[i]['stop_train'])+datetime.timedelta(days=7), color='green', lw=6, linestyle='dashed')  #str(data_path_lst[i]['color'])
    # if zoom:
        # axl[3+i].text(pd.to_datetime('2021-07-25 03:00:00'),0.75,data_path_lst[i]['label'], fontsize=30)
    # else:
        # axl[3+i].text(pd.to_datetime('2021-04-10'),0.75,data_path_lst[i]['label'], fontsize=30)
    
    
for i in range(len(data_path_lst)+3):
    axl[i].axvline(pd.to_datetime('2021-07-28 17:50:00'), color='red', lw=10, linestyle='dashed', alpha=0.5)  
    



for i in range(3, rows):
    y_ticks = [0,1]
    y_labels = ['Normal', 'Abnormal']
    axl[i].set_yticks(y_ticks)
    axl[i].set_yticklabels(y_labels)
    legendi = axl[i].legend(fontsize=legend_fontsize, loc="lower left")
    legendi.get_frame().set_alpha(0.3)



    
    

for i in range(rows):
    axl[i].tick_params(axis='both', which='major', labelsize=ylabel_fontsize)
    axl[i].grid()
    if zoom:
        axl[i].set_xlim(start, stop)




axl[rows-1].tick_params(axis='x', rotation=90, labelsize=ylabel_fontsize)
fig.subplots_adjust(top=0.95) 
fig.subplots_adjust(hspace=0.1)
# plt.savefig(fig_name+'.jpg', bbox_inches = 'tight', pad_inches = 0.2, dpi=100)
# plt.savefig(fig_name+'.pdf', bbox_inches = 'tight', pad_inches = 0.2, dpi=100)
plt.show()