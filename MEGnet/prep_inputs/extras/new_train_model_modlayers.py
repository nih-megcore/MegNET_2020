#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 11:49:16 2022

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
from sklearn import preprocessing
from matplotlib import pyplot as plt 
from sklearn.metrics import confusion_matrix

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('-output_dir')
    parser.add_argument('-normalize',
                        action='store_true',
                        help='Apply min/max normalization')
    parser.add_argument('-weights',
                        default=3,
                        help='Minority weighting for loss calc')
    args = parser.parse_args()
    output_dir = args.output_dir
    if args.normalize==True:
        NORMALIZE=True
    else:
       NORMALIZE=False
    w_ = int(args.weights)
    class_weights={0:1, 1:w_, 2:w_, 3:w_}
    if not os.path.exists(output_dir): os.mkdir(output_dir)


tmp=MEGnet.__path__[0]
class_table_path = op.join(tmp, 'prep_inputs', 'training', 'ICA_combined_participants.tsv')
class_table = pd.read_csv(class_table_path, sep='\t')
if 'Unnamed: 0' in class_table.columns:
    class_table.drop('Unnamed: 0', axis=1, inplace=True)
if 'idx' in class_table.columns:
    class_table.drop('idx', axis=1, inplace=True)



dataset_path = op.join(MEGnet.__path__[0], 'prep_inputs','training','ICAs')
dsets = glob.glob(op.join(dataset_path, '*_meg'))
dsets += glob.glob(op.join(dataset_path, '*-sss'))
dsets += glob.glob(op.join(dataset_path, '*_wrkmem'))
dsets += glob.glob(op.join(dataset_path, '*_rest'))
dsets += glob.glob(op.join(dataset_path, '*_AD'))
dsets += glob.glob(op.join(dataset_path, '*_NY'))
datasets = pd.DataFrame(dsets, columns=['dirname'])

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
    

datasets['subjid'] = datasets.dirname.apply(get_subjid)
datasets['type'] = datasets.dirname.apply(get_type)
datasets['type'] = datasets.type.apply(clean_KIT)

final = pd.merge(class_table, datasets, left_on=['participant_id', 'TaskType'], right_on=['subjid','type'])
dropidx=final.loc[(final.participant_id=='sub-ON12688') & (final.type_y =='rest')].index
final = final.drop(index=dropidx)
dropidx = final[final.TaskType=='artifact'].index
final = final.drop(index=dropidx)
final.reset_index(inplace=True, drop=True)

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
    assert spat_.shape == (20,180, 150, 3)
    
    spat_resized = spat_[:,25:-35,16:-14,:]
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

def test_fPredict():
    arrTimeSeries, arrSpatialMap = get_default_hcp()
    output = fPredictChunkAndVoting(kModel, 
                                    arrTimeSeries, 
                                    arrSpatialMap, 
                                    np.zeros((20,3)), #the code expects the Y values as it was used for performance, just put in zeros as a place holder.
                                    intModelLen=15000, 
                                    intOverlap=3750)
    arrPredicionsVote, arrGTVote, arrPredictionsChunk, arrGTChunk = output
    correct_out = [2, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 2, 0, 0]
    actual_out = arrPredicionsVote[:,0,:].argmax(axis=1)
    assert_vals = [np.equal(i,j) for i,j in zip(correct_out, actual_out)]
    assert False not in assert_vals


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
        # if TS_tmp.shape[1] < 62750:
            failed.append(input_vec) #continue
        TS_test.append(TS_tmp[:,:15000]) #62750])
        SP_test.append(SP_tmp)
        class_vec.append(CLid_tmp) 
        for i in failed:
            print(i)
    return np.vstack(TS_test), np.vstack(SP_test), np.stack(class_vec).flatten()



    
train_dir = op.join(MEGnet.__path__[0], 'prep_inputs','training')
np_arr_topdir = op.join(train_dir, 'Inputs')
arrTS_fname = op.join(np_arr_topdir, 'arrTS.npy')
arrSP_fname = op.join(np_arr_topdir, 'arrSP.npy')
arrC_ID_fname = op.join(np_arr_topdir, 'arrC_ID.npy')

arrTS_fname_45000 = op.join(np_arr_topdir, 'arrTS_45000.npy')
arrSP_fname_45000 = op.join(np_arr_topdir, 'arrSP_45000.npy')
arrC_ID_fname_45000 = op.join(np_arr_topdir, 'arrC_ID.npy')

def load_all_inputs(multistep=False):
    '''
    Returns the precompiled numpy arrays.

    Parameters
    ----------
    multistep : BOOL, optional
        If True, returns the 45000 second data. The default is False.

    Returns
    -------
    arrTimeSeries : np.ndarray
        Components X time - (15000 or 45000 designated by multistep)
    arrSpatialMap : np.ndarray
        Components X 256 X 256 X 3.
    class_ID : TYPE
        Component Classification.

    '''
    if multistep==False:
        arrTimeSeries = np.load(arrTS_fname)
        arrSpatialMap = np.load(arrSP_fname)
        class_ID = np.load(arrC_ID_fname)  
        assert arrTimeSeries.shape[0] == arrSpatialMap.shape[0]
        assert class_ID.shape[0] == arrTimeSeries.shape[0]
        assert final.__len__() == int(arrTimeSeries.shape[0]/20)
        print('Returning data with 15000 seconds')
    else:
        arrTimeSeries = np.load(arrTS_fname_45000)
        arrSpatialMap = np.load(arrSP_fname_45000)
        class_ID = np.load(arrC_ID_fname_45000)
        assert arrTimeSeries.shape[0] == arrSpatialMap.shape[0]
        assert class_ID.shape[0] == arrTimeSeries.shape[0]
        assert final.__len__() == int(arrTimeSeries.shape[0]/20)
        print('Returning data with 45000 seconds')
    return arrTimeSeries, arrSpatialMap, class_ID    



from tensorflow import keras
# import tensorflow_addons as tfa
# model_fname = op.join(MEGnet.__path__[0], 'model/MEGnet_final_model.h5')
# kModel = keras.models.load_model(model_fname, compile=False)

model_fname = op.join(MEGnet.__path__[0], 'model_extended2')
lModel = keras.models.load_model(model_fname, compile=False)
for idx,layer in enumerate(lModel.layers[:-2]):
    if idx in [0,1,2,3,10,11,12,13]:
        layer.trainable=True
    else:
        layer.trainable=False
    
kModel=lModel
# arrTimeSeries, arrSpatialMap, class_ID = load_all_inputs()

from tensorflow_addons.metrics import F1Score
f1_score=F1Score(4, average='macro')

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
# tmp_holdout = cvSplits.main(kfolds=5, foldNormFields=crossval_cols, data_dframe=final)
# holdout_dframe_idxs = tmp_holdout[0]['test_indx']  #First CV test set ~20% of data
# hold, tsttr = get_cv_npyArr(sample=None,
#                                           holdout=holdout_dframe_idxs,
#                                             arrTimeSeries=arrTimeSeries, 
#                                             arrSpatialMap=arrSpatialMap,
#                                             class_ID=class_ID
#                                             )
# hold_sp, hold_ts, hold_clID = hold['sp'], hold['ts'], hold['clID']
# tsttr_sp, tsttr_ts, tsttr_clID = tsttr['sp'], tsttr['ts'], tsttr['clID']




import sklearn



NB_EPOCH = 200
BATCH_SIZE = 700  
VERBOSE = 1
# OPTIMIZER = Adam()  #switch to AdamW
# VALIDATION_SPLIT = 0.20


kModel.compile(
    loss=keras.losses.CategoricalCrossentropy(), 
    optimizer=keras.optimizers.Adam(learning_rate=1e-4), 
    metrics=[f1_score, 'accuracy']
    )


import tensorflow as tf
from tensorflow import one_hot
earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                 patience=10,
                                 restore_best_weights=True)

## Load all of the cross validation data
inputs_dir = op.join(MEGnet.__path__[0], 'prep_inputs/training/Inputs')
tt_final = pd.read_csv(op.join(inputs_dir, 'TestTrain.csv'))
tsttr_ts = np.load(op.join(inputs_dir, 'tsttr_ts.npy'))
tsttr_sp = np.load(op.join(inputs_dir, 'tsttr_sp.npy'))
tsttr_clID = np.load(op.join(inputs_dir, 'tsttr_clID.npy'))
# tsttr_clID = one_hot(tsttr_clID, 4)

hold_ts = np.load(op.join(inputs_dir, 'hold_ts.npy'))
hold_sp = np.load(op.join(inputs_dir, 'hold_sp.npy'))
hold_clID = np.load(op.join(inputs_dir, 'hold_clID.npy'))
# hold_clID = one_hot(hold_clID, 4)

# if NORMALIZE==True:
#     print('Normalizing the data')
#     ## Normalize the data
#     #MinMax timeseries
#     mm_scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
#     tsttr_ts = mm_scaler.fit_transform(tsttr_ts.T).T
#     hold_ts = mm_scaler.fit_transform(hold_ts.T).T
    
#     #MinMax Spatial
#     mm_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
#     tmp = tsttr_sp.reshape([tsttr_sp.shape[0],-1])
#     tsttr_sp = mm_scaler.fit_transform(tmp.T).T.reshape(tsttr_sp.shape).astype(np.float16) #memory issues with float64
    
#     tmp = hold_sp.reshape([hold_sp.shape[0],-1])
#     hold_sp = mm_scaler.fit_transform(tmp.T).T.reshape(hold_sp.shape).astype(np.float16)





history=[]
with open(op.join(inputs_dir, 'CVdict.pkl'), 'rb') as f:
    cv = pickle.load(f)

# =============================================================================
# Burnin trainable layers
# =============================================================================
kModel.compile(
    loss=keras.losses.CategoricalCrossentropy(), 
    optimizer=keras.optimizers.Adam(), 
    metrics=[f1_score, 'accuracy']
    )

sample = cv[0]
tr, te = get_cv_npyArr(sample,
                      holdout=None,
                      arrTimeSeries=tsttr_ts,  #Subsampled array
                      arrSpatialMap=tsttr_sp, #Subsampled array
                      class_ID=tsttr_clID,  #Subsampled array
                    )

SP_, TS_, CL_   =  tr['sp'],tr['ts'], tr['clID']
               
history_tmp = kModel.fit(x=dict(spatial_input=SP_, temporal_input=TS_), y=one_hot(CL_,4),
                     batch_size=BATCH_SIZE, epochs=NB_EPOCH, verbose=VERBOSE,  
                     validation_data=(dict(spatial_input=te['sp'], temporal_input=te['ts']), one_hot(te['clID'],4)),
                     class_weight=class_weights, callbacks=[earlystop])

# =============================================================================
# 
# =============================================================================
for layer in kModel.layers:
    layer.trainable=True

kModel.compile(
    loss=keras.losses.CategoricalCrossentropy(), 
    optimizer=keras.optimizers.Adam(learning_rate=1e-5), 
    metrics=[f1_score, 'accuracy']
    )

# =============================================================================
# Continue training
# =============================================================================



for cv_num in cv.keys():
    sample = cv[cv_num]
    tr, te = get_cv_npyArr(sample,
                          holdout=None,
                          arrTimeSeries=tsttr_ts,  #Subsampled array
                          arrSpatialMap=tsttr_sp, #Subsampled array
                          class_ID=tsttr_clID,  #Subsampled array
                        )
    
    SP_, TS_, CL_   =  tr['sp'],tr['ts'], tr['clID']
                   
    history_tmp = kModel.fit(x=dict(spatial_input=SP_, temporal_input=TS_), y=one_hot(CL_,4),
                         batch_size=BATCH_SIZE, epochs=NB_EPOCH, verbose=VERBOSE,  
                         validation_data=(dict(spatial_input=te['sp'], temporal_input=te['ts']), one_hot(te['clID'],4)),
                         class_weight=class_weights, callbacks=[earlystop])
    history.append(history_tmp)

#%% 


def save_weights_and_history(history):
    for idx,epoch in enumerate(history):
        epo_dir = op.join(output_dir, f'epoch{idx}')
        os.mkdir(epo_dir)
        with open(f'{epo_dir}/trainHistoryDict', 'wb') as file_pi:
            pickle.dump(epoch.history, file_pi)

save_weights_and_history(history)
kModel.save(f'{output_dir}/model')

score = kModel.evaluate(x=dict(spatial_input=hold_sp, temporal_input=hold_ts), y=one_hot(hold_clID,4))
with open(f'{output_dir}/score', 'wb') as f:
    pickle.dump(score, f)
    
# =============================================================================
#     Plot and save history
# =============================================================================

i=0;  j=0
fig, axes = plt.subplots(3,3)
for epo in range(8):
    with open(f'{output_dir}/epoch{str(epo)}/trainHistoryDict', mode='rb') as w:
        history = pickle.load(w)
    if i==3:
        i=0
        j+=1
    axes[j,i].plot(history['accuracy'], 'r')
    axes[j,i].plot(history['f1_score'], 'g')    
    axes[j,i].plot(history['val_accuracy'], 'b+')
    axes[j,i].plot(history['val_f1_score'], 'k+')
    axes[j,i].set_title(f'Epoch{str(epo)}')
    i+=1
fig.suptitle('TrAcc(red) TeAcc(blue) TrF1(green) TeF1(black)')
fig.tight_layout()
fig.savefig('TestTrain_graph.png', dpi=300) 


y_hat = kModel.predict(x=dict(spatial_input=hold_sp, temporal_input=hold_ts))
y_pred = y_hat.argmax(axis=1)
matrix = confusion_matrix(hold_clID, y_hat.argmax(axis=1))
np.save(f'{output_dir}/confusion_mat.npy', matrix)
# =============================================================================
# 
# =============================================================================
# score = kModel.evaluate(x=dict(spatial_input=arrSpatialMap, temporal_input=arrTimeSeries), y=class_ID)
# score = kModel.evaluate(x=dict(spatial_input=hold_sp, temporal_input=hold_ts), y=hold_clID)    
    
# from matplotlib import pyplot as plt    
# for i in range(0,10):
#     # i=0
#     plt.plot(history[i].history['accuracy'])    
#     plt.plot(history[i].history['val_accuracy'])
#     plt.plot(history[i].history['get_f1'])


# =============================================================================
# 
# =============================================================================

# sc_=[]
# for i in range(0,10):
#     tmp_ = history[i].model.evaluate(x=dict(spatial_input=hold_sp, temporal_input=hold_ts), y=hold_clID)
#     sc_.append(tmp_)

# y_hat = kModel.predict(x=dict(spatial_input=hold_sp, temporal_input=hold_ts))
# y_pred = y_hat.argmax(axis=1)



# from sklearn.metrics import confusion_matrix
# matrix = confusion_matrix(hold_clID, y_hat.argmax(axis=1))


# from MEGnet.megnet_utilities import fPredictChunkAndVoting
# output = fPredictChunkAndVoting(kModel, hold_ts, hold_sp, np.zeros((hold_ts.shape[0], 3)))
# y_chunk_pred = output[0].argmax(axis=2).squeeze()

# # matrix = confusion_matrix(hold_clID, y_chunk_pred)

# output2 = fPredictChunkAndVoting(lModel, hold_ts[:,:15000], hold_sp, np.zeros((hold_ts.shape[0], 3)))
# matrix2 = confusion_matrix(hold_clID, output2[0].argmax(axis=2).squeeze())


