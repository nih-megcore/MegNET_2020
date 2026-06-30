#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 12 16:03:30 2023

@author: jstout
"""

import pandas as pd
import MEGnet
from os import path as op
import os
import glob
from scipy.io import loadmat
import numpy as np
import copy
import pickle


def get_subjid(dirname):
    tmp = op.basename(dirname)
    tmp=tmp.split('_')[0]
    if tmp[0:4]!='sub-':
        tmp='sub-'+tmp
    return tmp

def get_type(dirname):
    '''Extract the task type from the dataset name'''
    if 'task' in dirname:
        return [i[5:] for i in op.basename(dirname).split('_') if i[0:4]=='task'][0]
    else:
        return op.basename(dirname).split('_')[-1]

def clean_KIT(tasktype):
    if tasktype in  ['AD','NY']:
        return 'rest'
    else:
        return tasktype

# =============================================================================
# Only run if needed otherwise just load csv file from pkl dir    
# =============================================================================
# tmp=MEGnet.__path__[0]


# dframe = pd.read_csv(op.join(tmp, 'prep_inputs', 'training','Inputs','NIH_CAM_HCP_62750','Final_3site.csv'))
# dframe = dframe.loc[::20]  #This is the longform dataframe - get back to subject level
# dframe.drop_duplicates(['participant_id', 'Site', 'TaskType'], inplace=True)
# dframe.drop(['idx','Unnamed: 0.1', 'Unnamed: 0', 'ICAnum','ClassID','key'], axis=1, inplace=True)
# final = dframe

# final.drop_duplicates(['Site','participant_id','TaskType'], inplace=True)
# final.reset_index(inplace=True, drop=True)
# final['key']=range(len(final)) #Set the key to index data

final = pd.read_csv('/vf/users/EnigmaMeg/BIDS/ICA/ICA_megnet/PKL/dframe_TESTTRAIN.csv')
assert 'Glasgow' not in final.SheetName.unique()
assert 'REMPE' not in final.SheetName.unique()

def get_inputs(dataset_info):
    '''
    Load the MNE created datasets for each line of the dataframe corresponding
    to a single MEG acquisition

    Parameters
    ----------
    subj_info : pandas.series
        A line from the fully merged dataframe.

    Returns
    -------
    ts_ : numpy.ndarray
        ICA timeseries - ICA# X Samples
    spat_resized : numpy.ndarray
        ICA Spatial topography map - cropped by bounding box. 
        Shape of (20, 120, 120, 3)  -- (ICA, X, Y, Color)

    '''
    data_dir = dataset_info['dirname']
    ts_fname = op.join(data_dir, 'ICATimeSeries.mat')
    ts_ = loadmat(ts_fname)['arrICATimeSeries'].T
    assert type(ts_) is np.ndarray
    
    spat_ = [] 
    for i in range(1,21): spat_.append(loadmat(op.join(data_dir, f'component{str(i)}.mat'))['array'])
    assert len(spat_)==20
    spat_ = np.stack(spat_)
    if spat_.shape != (20,120, 120, 3):
        spat_ =  spat_ [:,25:-35,16:-14,:]
    assert spat_.shape == (20,120, 120, 3)
    
    spat_resized = spat_ #[:,25:-35,16:-14,:]
    class_vec = make_classification_vector(dataset_info)
    
    return ts_ , spat_resized, class_vec

def get_default_hcp():
    '''Load and return the hcp ICA dataset'''
    data_dir = op.join(MEGnet.__path__[0], 'example_data/HCP/100307/@rawc_rfDC_8-StoryM_resample_notch_band/ICA202DDisc')
    ts_ = loadmat(op.join(data_dir, 'ICATimeSeries.mat'))['arrICATimeSeries'].T
    
    # lSpatial.append(loadmat(os.path.join(strDataPathSpatial,f'component{intComp}.mat'))['array'][30:-30,15:-14,:])
    spat_ = []
    for i in range(1,21): spat_.append(loadmat(op.join(data_dir, f'component{str(i)}.mat'))['array'][30:-30,15:-15,:])
    spat_ = np.stack(spat_)
    return ts_, spat_


def _convert_strlist2intlist(strlist):
    '''Hack to fix formatting'''
    tmp_ = strlist.replace('"','').replace('[','').replace(' ','').replace(']','').replace("'","").split(',')
    if (tmp_=='') | (tmp_==[]) | (tmp_==['']):
        return []
    return [int(i) for i in tmp_]
    

def make_classification_vector(input_vec):
    '''Convert the separate labelled columns into a 20X1 vector of labels'''
    output = np.zeros(20, dtype=int) #Number of ICAs
    VEOG =  _convert_strlist2intlist(input_vec.eyeblink)
    HEOG = _convert_strlist2intlist(input_vec.Saccade)
    EKG = _convert_strlist2intlist(input_vec.EKG)
    output[VEOG] = 1 
    output[HEOG] = 3
    output[EKG] = 2
    return output

def extract_all_datasets(dframe):
    '''
    Loop over all datasets
    Load the spatial, temporal, and classIDs into numpy matrcies

    Parameters
    ----------
    dframe : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    TS_test, SP_test, class_vec = [], [], []
    for idx,input_vec in dframe.iterrows():
        print(idx)
        print(input_vec)
        TS_tmp, SP_tmp, CLid_tmp = get_inputs(input_vec)
        failed=[]
        if TS_tmp.shape[1] < 40000: 
            failed.append(input_vec) 
        TS_test.append(TS_tmp[:,:45000]) 
        
        SP_test.append(SP_tmp)
        class_vec.append(CLid_tmp) 
        for i in failed:
            print(i)
    return np.vstack(TS_test), np.vstack(SP_test), np.stack(class_vec).flatten()

def extract_and_pickle_all_datasets(dframe, pkl_dir):
    '''
    Loop over all datasets
    Load the spatial, temporal, and classIDs into numpy matrcies

    Parameters
    ----------
    dframe : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    TS_test, SP_test, class_vec = [], [], []
    for idx,input_vec in dframe.iterrows():
        print(idx)
        print(input_vec)
        TS_tmp, SP_tmp, CLid_tmp = get_inputs(input_vec)
        outdir = op.join(pkl_dir, f'{input_vec.Site}_{input_vec.subjid}_{input_vec.TaskType}')
        os.mkdir(outdir)
        ts_pklfile = op.join(outdir, 'ts.pkl')
        sp_file = op.join(outdir, 'sp.npy')
        cl_file = op.join(outdir, 'cl.npy')
        
        with open(ts_pklfile,'wb') as f:
            pickle.dump(TS_tmp, f)
        np.save(sp_file, SP_tmp)
        np.save(cl_file, CLid_tmp)
        del TS_tmp, SP_tmp, CLid_tmp, ts_pklfile, sp_file, cl_file
        

# os.mkdir(pkl_topdir)
pkl_topdir = '/data/EnigmaMeg/BIDS/ICA/ICA_megnet/PKL'
extract_and_pickle_all_datasets(final, pkl_topdir)


# =============================================================================
# 
# =============================================================================
pkl_dir = '/vf/users/EnigmaMeg/BIDS/ICA/ICA_megnet/PKL' # '/home/jstout/src/MegNET2022/MEGnet/prep_inputs/training/Inputs/PKL'
pkl_topdir = pkl_dir
# dframe_hold = pd.read_csv(op.join(pkl_topdir, 'dframe_holdout.csv'))
dframe_train = pd.read_csv(op.join(pkl_topdir, 'dframe_TESTTRAIN.csv'))
assert 'Glasgow' not in dframe_train.SheetName.unique()
assert 'REMPE' not in dframe_train.SheetName.unique()



# # =============================================================================
# # Holdout
# # =============================================================================
# holds_fname = op.join(pkl_topdir, 'hold_data.pkl')
# if not op.exists(holds_fname):    
#     hold_list = []
#     for key in dframe_hold.key.to_list():
#         hold_list.append(subj_stack[key])
    
#     holds_fname = op.join(pkl_topdir, 'hold_data.pkl')    
#     with open(holds_fname, 'wb') as f:
#         pickle.dump(hold_list, f)
# else:
#     with open(holds_fname, 'rb') as f: hold_list=pickle.load(f)

# # =============================================================================
# # Train data
# # =============================================================================
# train_fname=op.join(pkl_topdir, 'train_data.pkl') 
# if not op.exists(train_fname):
#     train_list = []
#     for key in dframe_train.key.to_list():
#         train_list.append(subj_stack[key])
    
#     with open(train_fname, 'wb') as f:
#         pickle.dump(train_list, f)
# else:
#     with open(train_fname, 'rb') as f: train_list = pickle.load(f)






# =============================================================================
# Cross Val is already part of the data frame
# =============================================================================


# =============================================================================
# Cross Validation
# =============================================================================
crossval_cols = ['Site', 'TaskType', 'Scanner', 'age', 'sex']
from MEGnet.prep_inputs import cvSplits
final['idx']=final.index

#Use the following function to match the CV to the ICAs
def make_ica_subj_encoding(arrTimeSeries):
    '''Expand the coding for each subject by 20 - to match the number of ICAs'''
    lenval = arrTimeSeries.shape[0]
    idxs = range(lenval)
    test = [[i]*20 for i in range(int(lenval/20))]
    test = np.hstack(test)
    assert len(test) == len(idxs)
    return np.array([idxs, test]).T

def get_cv_npyArr(sample=None,
                  holdout=None,
                    arrTimeSeries=None, 
                    arrSpatialMap=None,
                    class_ID=None
                    ):
    '''
    Return the numpy array for the test / train slice
    

    Parameters
    ----------
    sample : array of ints
        Cross validation sample of the dataframe indexes.
    outcode : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''

    
    #ICA number is in column 0 and subject index is column2
    ica_code = make_ica_subj_encoding(arrTimeSeries)
    
    if holdout is None:
        cv_tr = sample['train_indx']
        cv_te = sample['test_indx']
        
        #Probably slow - but will work
        tr_idx = [ica_code[ica_code[:,1]==i] for i in cv_tr]
        tr_idx = np.vstack(tr_idx)
        
        te_idx = [ica_code[ica_code[:,1]==i] for i in cv_te]
        te_idx = np.vstack(te_idx)
        
        #Subsample the cv
        tr_sub = ica_code[tr_idx[:,0],0]
        te_sub = ica_code[te_idx[:,0],0]
        train={'sp':arrSpatialMap[tr_sub,:,:,:],
                   'ts':arrTimeSeries[tr_sub,:],
                   'clID':class_ID[tr_sub]}
        test={'sp':arrSpatialMap[te_sub,:,:,:],
                   'ts':arrTimeSeries[te_sub,:],
                   'clID':class_ID[te_sub]}
        return train, test
    else:
        cv_hold = holdout
        hold_idx = [ica_code[ica_code[:,1]==i] for i in cv_hold]
        hold_idx = np.vstack(hold_idx)
        hold_sub = ica_code[hold_idx[:,0],0]
        
        #tt_array is the test/train array - excluding the holdout
        tmp_full_array = copy.deepcopy(ica_code)
        tt_array = np.delete(tmp_full_array, hold_sub, axis=0)
        # not_hold_idx = [ica_code[ica_code[:,1]==i] for i in tt_array]
        # not_hold_idx = 
        tt_sub = tt_array[:,0]
        
        hold={'sp':arrSpatialMap[hold_sub,:,:,:],
              'ts':arrTimeSeries[hold_sub,:],
              'clID':class_ID[hold_sub]}
        test_train={'sp':arrSpatialMap[tt_sub,:,:,:],
               'ts':arrTimeSeries[tt_sub,:],
               'clID':class_ID[tt_sub]}
        return hold, test_train
        
#Create holdout
tmp_holdout = cvSplits.main(kfolds=5, foldNormFields=crossval_cols, data_dframe=dframe_train)
holdout_dframe_idxs = tmp_holdout[0]['test_indx']  #First CV test set ~20% of data
final['HoldOut']=False
final.loc[holdout_dframe_idxs, 'HoldOut']=True

#Create TestTrain
tmp_holdout[0]['train_indx']
tsttr_dframe=final[final['HoldOut']==False]


hold, tsttr = get_cv_npyArr(sample=None,
                                          holdout=holdout_dframe_idxs,
                                            arrTimeSeries=arrTimeSeries, 
                                            arrSpatialMap=arrSpatialMap,
                                            class_ID=class_ID
                                            )
hold_sp, hold_ts, hold_clID = hold['sp'], hold['ts'], hold['clID']
tsttr_sp, tsttr_ts, tsttr_clID = tsttr['sp'], tsttr['ts'], tsttr['clID']

# =============================================================================
# Create testing CV splits
# =============================================================================
dframe_train['idx']=range(len(dframe_train))
tt_final =  dframe_train            #final.drop(index=holdout_dframe_idxs)
tt_final.reset_index(inplace=True, drop=True)
cv = cvSplits.main(kfolds=6, foldNormFields=crossval_cols, data_dframe=tt_final)
for cv_num in cv.keys():
    tt_final[f'Fold{str(cv_num)}']=False
    
    #Get indexes from the subselected dataframe (basically removed holdout)
    sub_idx_train = cv[cv_num]['train_indx']
    sub_idx_test = cv[cv_num]['test_indx']
    
    #Get the index in the full dataframe
    tt_final_train_idx = tt_final.loc[sub_idx_train,'idx']
    tt_final_test_idx = tt_final.loc[sub_idx_test,'idx']
    
    #Initialize all
    tt_final[f'Fold{str(cv_num)}']='Holdout'
    tt_final.loc[tt_final_train_idx, f'Fold{str(cv_num)}'] = 'Train'
    tt_final.loc[tt_final_test_idx, f'Fold{str(cv_num)}'] ='Test'
    print(cv_num)

# Verify holdouts are not in train/test    !!!
fold_cols = final.columns[final.columns.str[0:4]=='Fold']
for idx,row in final[final.HoldOut==True].iterrows():
    for col_var in row[fold_cols]:
        assert col_var=='Holdout'
    print(idx)

# Verify train/test are not in holdouts
for idx,row in final[final.HoldOut!=True].iterrows():
    for col_var in row[fold_cols]:
        assert col_var!='Holdout'
    print(idx)    
    
    
# =============================================================================
# Create a dataframe the same lenght as the ClassIDs/SP/TS arrays
# =============================================================================
def make_longform_dframe(dframe, class_ID=None):
    '''
    Convert the subject level dataframe to component level dataframe.  Indexes
    will match the 0-axis/row of the numpy array.

    Parameters
    ----------
    dframe : pd.DataFrame
        Subject level dataframe (merged demographics and ICA info).
    classID : np.array
        Class IDs for the rows of ICAs.  Used as a sanity check (assert).

    Returns
    -------
    dframe_longform : pd.DataFrame
        ICA level dataframe.

    '''
    init_cols = dframe.columns    
    ica_dummy_var = [f'ICA0{str(i)}' for i in range(10)] +  [f'ICA{str(i)}' for i in range(10,20)]
    dframe_wideform = copy.deepcopy(dframe)
    for idx, row in dframe_wideform.iterrows():
        dframe_wideform.loc[idx,ica_dummy_var]=make_classification_vector(row)
    
    dframe_longform = pd.melt(dframe_wideform, id_vars=init_cols, value_vars=ica_dummy_var,
                             var_name='ICAnum', value_name='ClassID')
    
    dframe_longform.sort_values(by=['key','ICAnum'], ascending=True, inplace=True)
    dframe_longform.reset_index(drop=True, inplace=True)
    
    
    dframe_longform = dframe_longform.drop(columns=['eyeblink', 'Saccade', 'EKG', 'other', 'Unnamed: 7', 'Unnamed: 6', 'type_y','type_x'])
    dframe_longform.ClassID = dframe_longform.ClassID.astype(int)
    
    match_idxs = [i*20 for i in range(len(dframe))]
    tmp = dframe.participant_id.values == dframe_longform.loc[match_idxs, 'participant_id'].values
    assert False not in tmp
    assert np.alltrue(class_ID==dframe_longform.ClassID)
    return dframe_longform


final_longform = make_longform_dframe(tt_final, class_ID=class_ID)
out_fname = op.join(op.dirname(np_arr_topdir), 'final_longform_dframe.csv')
final_longform.to_csv(out_fname)

# =============================================================================
# Build time series chunks and normalize
# =============================================================================





# # =============================================================================
# # Quick VIS checks
# # =============================================================================
# veog = final_longform[final_longform.ClassID==1].index
# ekg = final_longform[final_longform.ClassID==2].index
# heog = final_longform[final_longform.ClassID==3].index

# from matplotlib import pyplot as plt
# i=0; j=0
# fig, axes = plt.subplots(5,2)
# for i in range(10):
#     k=copy.deepcopy(i)
#     if i>4:
#         k-=5
#     axes[i,j].plot(arrTimeSeries[ekg[i],:1000])
#     if i==4: #(i%4==0) and (i!=0):
#         j+=1
        

#     output[VEOG] = 1 
#     output[HEOG] = 3
#     output[EKG] = 2
    


# # =============================================================================
# # Test group diffs
# # =============================================================================
# min_ = arrTimeSeries.min(axis=1)
# max_ = arrTimeSeries.max(axis=1)
# min_max = pd.DataFrame(zip(min_, max_), columns=['min','max'])

# ts_info_dframe = pd.merge(final_longform, min_max, left_index=True, right_index=True)

# ts_info_dframe.groupby(['Site','ClassID'])['max'].agg([np.max,np.mean])
# ts_info_dframe.groupby(['Site','ClassID'])['min'].agg([np.min,np.mean])

# out_1 = ts_info_dframe.groupby(['Site'])['max'].agg([np.mean]).values - ts_info_dframe.groupby(['Site'])['min'].agg([np.mean]).values
# pd.DataFrame(out_1, index=['Cam','NIH','NYU','SLU'], columns=['Mean Range'])

# # =============================================================================
# # 
# # =============================================================================



