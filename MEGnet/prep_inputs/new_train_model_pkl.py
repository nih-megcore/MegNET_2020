#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 21:23:50 2023

@author: jstout
"""


import pandas as pd

from os import path as op
import os
import glob
from scipy.io import loadmat
import numpy as np
import copy
import pickle

from numpy.random import randint
from numpy.random import shuffle
import copy

import MEGnet
from MEGnet.megnet_utilities import fGetStartTimesOverlap
tmp=MEGnet.__path__[0]

train_dir = '/vf/users/EnigmaMeg/BIDS/ICA/ICA_megnet/TRAINING_DATA_despike'
pkl_topdir = '/vf/users/EnigmaMeg/BIDS/ICA/ICA_megnet/PKL'
os.chdir(pkl_topdir)
#train_dir = op.join(MEGnet.__path__[0], 'prep_inputs','training')
#pkl_topdir = op.join(train_dir, 'Inputs', 'PKL')


# os.mkdir(pkl_topdir)
# extract_and_pickle_all_datasets(final, pkl_topdir)
# dframe = pd.read_csv('/home/jstout/src/MegNET2022/MEGnet/prep_inputs/training/final_subjICA_dframe.csv')

# dframe_fname = op.join(pkl_topdir, 'dframe_train.csv') #'final_dframe.csv')
# dframe= pd.read_csv(dframe_fname)
# dframe = dframe[~ dframe.duplicated(['Site','TaskType', 'participant_id'])]

# dframe = dframe[dframe.Site != 'NYU']
# dframe.reset_index(inplace=True)
# #KEY indexes original subj_stack
# dframe.drop(['idx','index','Unnamed: 0.1', 'Unnamed: 0'], axis=1, inplace=True)

# =============================================================================
# Just load the proper dframe from pkl dir
# =============================================================================
dframe = pd.read_csv(op.join(pkl_topdir, 'final_TESTTRAIN.csv'))
assert 'REMPE' not in dframe.SheetName.unique()
assert 'Glasgow' not in dframe.SheetName.unique()
# positive assertion
assert 'CAMCAN_smt_mvcorr' in dframe.SheetName.unique()

# =============================================================================
# Assemble time series from the 
# =============================================================================

arrTS = []
arrSP = []
arrCL = []

dframe.key = range(len(dframe))
for idx, row in dframe.iterrows():
    input_vec = row
    indir = op.join(pkl_topdir, f'{input_vec.Site}_{input_vec.subjid}_{input_vec.TaskType}')
    print(indir)
    with open(op.join(indir, 'ts.pkl'),'rb') as f:
        ts = pickle.load(f)
    sp_fname = op.join(indir, 'sp.npy')
    cl_fname = op.join(indir, 'cl.npy')
    sp = np.load(sp_fname)
    cl = np.load(cl_fname)
    arrTS.append(ts)
    arrSP.append(sp)
    arrCL.append(cl)

assert len(arrTS) == len(dframe)
    
def prep_ts_blocks(arrTimeSeries, overlap=3750, normalize=True):
    '''Cut the data into 1min blocks and temporally normalize each block
    Currently assuming all datasets are the same number of samples'''
    # Chunk Data
    # num_blocks = 1 + (arrTimeSeries.shape[1] - 15000) // overlap
    # starts = np.arange(num_blocks)*3750
    starts = fGetStartTimesOverlap(arrTimeSeries.shape[1])
    blocks = np.zeros([arrTimeSeries.shape[0], 15000, len(starts)], dtype=float)
    for i,start in enumerate(starts): 
        blocks[:,:, i] = copy.deepcopy(arrTimeSeries[:,start:start+15000]) 
    
    multiplier = 10
    stds=np.std(blocks,axis=1)
    maxes=np.max(blocks,axis=1)    
    blocks_with_spikes = np.unique(np.where(maxes > stds*multiplier)[1])
    blocks = np.delete(blocks, blocks_with_spikes, axis=-1)
    
    if normalize==True:
        testTimeSeries = blocks #copy.deepcopy(arrScanTimeSeries[intStartTime:intStartTime+intModelLen])
        min_vals = np.min(testTimeSeries, axis=1, keepdims=True)
        max_vals = np.max(testTimeSeries, axis=1, keepdims=True)
        scaling_factors = 8 / (max_vals - min_vals)
        mean_vals = np.mean(testTimeSeries, axis=1, keepdims=True)
        testTimeSeries = testTimeSeries - mean_vals
        testTimeSeries = testTimeSeries * scaling_factors 
    return testTimeSeries

subj_stack = {}
drop_idx=[]
for idx, _ in enumerate(arrTS): 
    print(idx)
    current_TS = arrTS[idx]
    current_sp = arrSP[idx]
    current_cl = arrCL[idx]
    #Chop temporal series into blocks
    ts_tmp = prep_ts_blocks(current_TS)
    if ts_tmp.shape[2]==0:
        drop_idx.append(dframe.loc[idx,'FILENAME'])
        continue
    ts_out = np.vstack([ ts_tmp[:,:,i] for i in range(ts_tmp.shape[2])])
    repeats = ts_tmp.shape[2]
    #Repeat the spatial maps and class    
    sp_out = np.vstack([ current_sp for i in range(repeats) ]) 
    cl_out = np.tile(current_cl, repeats)
    input_series = dframe.loc[idx]
    subj_stack[idx] = {'ts':ts_out,
                       'sp':sp_out,
                       'cl':cl_out,
                       'id':f'{input_series.Site}_{input_series.subjid}_{input_series.TaskType}'
                       }

assert len(subj_stack)==len(dframe)-len(drop_idx)



del arrTS, arrSP, arrCL

def make_shuffled_nparrs(dframe_in, subj_stack):
    '''This code takes the full subject list w/pkl data and returns arrays'''
    idxs = dframe_in.key.to_list()
    shuffle(idxs)
    dframe_in['Shuffle_idx']=idxs
    dframe_in.sort_values(['Shuffle_idx'], inplace=True)
    
    shuff_ts = []
    shuff_sp = []
    shuff_cl = []
    
    for idx in dframe_in.key.to_list():
        shuff_ts.append(subj_stack[idx]['ts'])
        shuff_sp.append(subj_stack[idx]['sp'])
        shuff_cl.append(subj_stack[idx]['cl'])
    
    arrTS = np.vstack(shuff_ts)
    arrSP = np.vstack(shuff_sp)
    arrCL = np.hstack(shuff_cl)
    return arrTS, arrSP, arrCL

dframe_train=pd.DataFrame(dframe[dframe.Fold0=='Train'])
dframe_test=pd.DataFrame(dframe[dframe.Fold0=='Test'])

train_drop_idxs = dframe_train.query(f'FILENAME in {drop_idx}').index
dframe_train.drop(train_drop_idxs, inplace=True)

test_drop_idxs = dframe_test.query(f'FILENAME in {drop_idx}').index
dframe_test.drop(test_drop_idxs, inplace=True)


tr_arrTS, tr_arrSP, tr_arrCL = make_shuffled_nparrs(dframe_train, subj_stack) 
te_arrTS, te_arrSP, te_arrCL = make_shuffled_nparrs(dframe_test, subj_stack) 

#Holdout has already been processed and saved out as pickle data
# hd_arrTS, hd_arrSP, hd_arrCL = make_shuffled_nparrs(dframe_hold, subj_stack)

out_topdir = op.join(train_dir, 'train_w_rej10')
if not op.exists(out_topdir): os.mkdir(out_topdir)
np.save(op.join(out_topdir,'tr_arrTS.npy'), tr_arrTS)
np.save(op.join(out_topdir,'tr_arrSP.npy'), tr_arrSP)
np.save(op.join(out_topdir,'tr_arrCL.npy'), tr_arrCL)

out_topdir = op.join(train_dir, 'test_w_rej10')
if not op.exists(out_topdir): os.mkdir(out_topdir)
np.save(op.join(out_topdir,'te_arrTS.npy'), te_arrTS)
np.save(op.join(out_topdir,'te_arrSP.npy'), te_arrSP)
np.save(op.join(out_topdir,'te_arrCL.npy'), te_arrCL)


# =============================================================================
# Load the already saved arrays
# =============================================================================
out_topdir = '/data/EnigmaMeg/MEGNET/KEEP_TEST_NPY2/FULL_nparr/train'
#op.join(train_dir, 'Inputs', 'FULL_nparr', 'train')
tr_arrTS = np.load(op.join(out_topdir,'tr_arrTS.npy'))
tr_arrSP = np.load(op.join(out_topdir,'tr_arrSP.npy'))
tr_arrCL = np.load(op.join(out_topdir, 'tr_arrCL.npy'))
out_topdir = '/data/EnigmaMeg/MEGNET/KEEP_TEST_NPY2/FULL_nparr/test'
te_arrTS = np.load(op.join(out_topdir,'te_arrTS.npy'))
te_arrSP = np.load(op.join(out_topdir,'te_arrSP.npy'))
te_arrCL = np.load(op.join(out_topdir, 'te_arrCL.npy'))



# =============================================================================
# 
# =============================================================================
assert tr_arrTS.shape[0]==tr_arrSP.shape[0]
assert tr_arrCL.shape[0]==tr_arrSP.shape[0]

# =============================================================================
# 
# =============================================================================
NB_EPOCH = 100
BATCH_SIZE = 300  
VERBOSE = 1

from sklearn.metrics import confusion_matrix
import sklearn
from tensorflow_addons.metrics import F1Score
import tensorflow as tf
from tensorflow import one_hot
from tensorflow import keras
from sklearn.utils.class_weight import compute_class_weight

clw = compute_class_weight('balanced', classes=np.array([0,1,2,3]), y=tr_arrCL)
clw/=clw.min()
class_weights={i:j for i,j in enumerate(clw)}

model_fname = op.join(MEGnet.__path__[0], 'model/MEGnet_final_model.h5')
kModel = keras.models.load_model(model_fname, compile=False)

# class_weights = {0:1, 1:15, 2:15, 3:15}
from numpy.random import uniform
for idx,layer in enumerate(kModel.layers):
    weights = layer.get_weights()
    if weights==[]:
        continue
    layer_rand_w=[]
    for sublayer in weights:
        layer_rand_w.append(uniform(low=-1.0, high=1.0, size=sublayer.shape))
    layer.set_weights(layer_rand_w)


f1_score=F1Score(4, average='macro')
kModel.compile(
    loss=keras.losses.CategoricalCrossentropy(), 
    optimizer=keras.optimizers.Adam(learning_rate=1e-3), 
    metrics=[f1_score, 'accuracy']
    )

earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_f1_score',
                                 patience=8,
                                 restore_best_weights=True)







def save_weights_and_history(history, kModel, cv_num):
    # for idx,epoch in enumerate(history):
    epo_dir = op.join(output_dir, f'epoch{str(cv_num)}')
    if not os.path.exists(epo_dir): os.mkdir(epo_dir)
    with open(f'{epo_dir}/history.pkl', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    kModel.save(f'{epo_dir}/model')
    # with open(f'{epo_dir}/score', 'wb') as file_sc:
    #     pickle.dump(score[idx], file_sc)

from MEGnet.prep_inputs.augmentation_generator import MEGnetAugSequence

train_seq = MEGnetAugSequence(
    arrSP=tr_arrSP, arrTS=tr_arrTS, arrCL=tr_arrCL,
    batch_size=BATCH_SIZE,
    noise_snr=0.1,      # tune: noise std as fraction of per-sample TS std
    p_flip=0.5,         # 50% coupled flip
    cutoff_hz=120.0,
    sfreq=250.0,
    num_classes=4,
    shuffle=True,
    seed=0,
)

history_tmp = kModel.fit(
    train_seq,
    epochs=NB_EPOCH,
    verbose=VERBOSE,
    validation_data=(dict(spatial_input=te_arrSP, temporal_input=te_arrTS),
                     one_hot(te_arrCL, 4)),
    class_weight=class_weights,
    callbacks=[earlystop],
)


              
#history_tmp = kModel.fit(x=dict(spatial_input=tr_arrSP, temporal_input=tr_arrTS), y=one_hot(tr_arrCL,4),
#                     batch_size=BATCH_SIZE, epochs=NB_EPOCH, verbose=VERBOSE,  
#                     validation_data=(dict(spatial_input=te_arrSP, temporal_input=te_arrTS), one_hot(te_arrCL,4)),
#                     class_weight=class_weights, callbacks=[earlystop])

epochs = range(len(history_tmp.history['f1_score']))
pylab.plot(epochs, history_tmp.history['f1_score'], '+', epochs, history_tmp.history['val_f1_score'],'.')
pylab.legend(['F1_train','F1_val'])

output_dir = op.join(MEGnet.__path__[0], 'model_nih')
save_weights_and_history(history_tmp, kModel, 0)



    
