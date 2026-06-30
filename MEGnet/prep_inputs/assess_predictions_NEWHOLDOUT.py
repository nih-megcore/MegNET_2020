#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 12:55:01 2023

@author: jstout
"""

import os, os.path as op
import glob
import copy
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from tensorflow import keras
import numpy as np
import pandas as pd
import pickle
import MEGnet
from MEGnet.megnet_utilities import fGetStartTimesOverlap
from MEGnet.megnet_utilities import fPredictChunkAndVoting_parrallel as fPredictChunkAndVoting
# import tensorflow_addons
from sklearn.metrics import confusion_matrix, f1_score
import sys
from scipy.io import loadmat

## Objectives
# Take a folder and run the classification on all ICAs found in the folder
# Generate a logfile of the outputs 

os.chdir('/vf/users/EnigmaMeg/BIDS/ICA/ICA_megnet')
#assess_dirs = ['NIH_yanovski','NIH_stringaris','ICA_output_glasgow','KIT_UMD_ICAs']
## Missing REMPE
#dframe_holdout_new.csv

pkl_dir = '/vf/users/EnigmaMeg/BIDS/ICA/ICA_megnet/PKL'
pkl_topdir = pkl_dir
dframe_hold = pd.read_csv(op.join(pkl_topdir, 'dframe_holdout_new.csv'))
dframe_hold['subjid']=dframe_hold.participant_id
####
def get_all_ica_dirs(input_dir):
    '''Locate all of the ICA folders
    Check for sub-*_ica and sub-*_meg folders
    
    returns ica_dirs :: list'''
    ica_dirs = glob.glob(op.join(input_dir, 'sub-*_ica'))
    if ica_dirs.__len__() == 0:
        ica_dirs = glob.glob(op.join(input_dir, 'sub-*_meg'))
    return ica_dirs

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

def get_inputs(ica_dir):
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
    data_dir = ica_dir
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




# =============================================================================
# Run the prediction scoring
# =============================================================================
import sys
tmp = sys.argv[1]
MODEL_TOPDIR='/data/EnigmaMeg/MEGNET/MODELFIT'
kModel_fname = op.join(MODEL_TOPDIR, tmp, 'model')
#kModel_fname = op.join(MEGnet.__path__[0], 'model_v2')
kModel = keras.models.load_model(kModel_fname, compile=False)

# =============================================================================
# Load in the data and process
# =============================================================================


## Load each pkl and numpy file for each dataset
holdout = {}
for idx, input_vec in dframe_hold.iterrows():
    outdir = op.join(pkl_dir, f'{input_vec.Site}_{input_vec.subjid}_{input_vec.TaskType}')
    print(outdir)
    ts_pklfile = op.join(outdir, 'ts.pkl')
    sp_file = op.join(outdir, 'sp.npy')
    cl_file = op.join(outdir, 'cl.npy')
    holdout[idx]={'ts':None, 'sp':None, 'cl':None}
    with open(ts_pklfile, 'rb') as f: holdout[idx]['ts'] = pickle.load(f)
    with open(sp_file, 'rb') as g: holdout[idx]['sp'] = np.load(g)
    with open(cl_file, 'rb') as h: holdout[idx]['cl'] = np.load(h)

## Assess the data using the keras model
for idx in holdout.keys():  #These are the keys to the dictionary not dataframe
    arrTS = holdout[idx]['ts']
    arrSP = holdout[idx]['sp']
    arrPredicionsVote, arrProb = fPredictChunkAndVoting(kModel, arrTS, arrSP,  
                                           intModelLen=15000, 
                                           intOverlap=3750)
    holdout[idx]['arrPred'] = arrPredicionsVote.argmax(axis=1)
    holdout[idx]['arrProb'] = arrProb


## Build out the stats from the assessments above
ypred = np.hstack([holdout[i]['arrPred'] for i in holdout.keys()])
cl = np.hstack([holdout[i]['cl'] for i in holdout.keys()])

conf_mat = confusion_matrix(cl, ypred)
acc = np.sum(cl==ypred)/len(cl)
ax1_sum = conf_mat.sum(axis=1)
diag_vals = copy.deepcopy(np.diag(conf_mat))
np.fill_diagonal(conf_mat, 0)
positive_pred = diag_vals / ax1_sum
false_neg = conf_mat.sum(axis=1)/ax1_sum
# false_pos = conf_mat.sum(axis=0)/ax1_sum
f1score = f1_score(cl, ypred, average='macro')

# print(f'######### {dirname} #########')
print(f'Accuracy:  {acc}')
print(f'PosPred: {positive_pred}')
print(f'FalseNeg: {false_neg}')
print(f'F1score: {f1score}')

holdout_fname = op.join(MODEL_TOPDIR, sys.argv[1], 'holdout.pkl')
with open(holdout_fname, 'wb') as f:
    pickle.dump(holdout, f)
dframe_hold.to_csv(op.join(sys.argv[1], 'holdout_dframe.csv'))
conf_mat.save(op.join(sys.argv1[1], 'confusion_mat.npy'))

# f.write(f'######### {dirname} #########\n')
# f.write(f'Accuracy:  {acc}\n')
# f.write(f'PosPred: {positive_pred}\n')
# f.write(f'FalseNeg: {false_neg}\n')
# f.write(f'F1score: {f1score}\n')


# =============================================================================
# Done
# =============================================================================
# epoch_eval={}    
# for dirname in glob.glob('epo*'): #epo_num in [0,1,2,3,4,5, 6]:
#     pred, prob = calc_stats(dirname)
#     epoch_eval[f'{dirname}'] = pred
#     epoch_eval[f'{dirname}_prob'] = prob
#     pred_fname = op.join(topdir, f'{dirname}', 'predictions.npy')
#     prob_fname = op.join(topdir, f'{dirname}', 'probabilities.npy')
#     np.save(pred_fname, epoch_eval[f'{dirname}'])
#     np.save(prob_fname, epoch_eval[f'{dirname}_prob'])
#     conf_mat = confusion_matrix(cl, pred)
#     np.save(op.join(topdir, f'{dirname}', 'confusion_mat.npy') ,conf_mat)

# os.chdir(topdir)
# out_fname = op.join(topdir, 'results.txt')
# with open(out_fname, 'w') as f:
#     for dirname in glob.glob('epo*'): #[0,1,2,3,4,5, 6]:
#         ypred = epoch_eval[f'{dirname}']
#         conf_mat = confusion_matrix(cl, ypred)
#         acc = np.sum(cl==ypred)/len(cl)
#         ax1_sum = conf_mat.sum(axis=1)
#         diag_vals = copy.deepcopy(np.diag(conf_mat))
#         np.fill_diagonal(conf_mat, 0)
#         positive_pred = diag_vals / ax1_sum
#         false_neg = conf_mat.sum(axis=1)/ax1_sum
#         # false_pos = conf_mat.sum(axis=0)/ax1_sum
#         f1score = f1_score(cl, ypred, average='macro')
        
#         print(f'######### {dirname} #########')
#         print(f'Accuracy:  {acc}')
#         print(f'PosPred: {positive_pred}')
#         print(f'FalseNeg: {false_neg}')
#         print(f'F1score: {f1score}')
#         f.write(f'######### {dirname} #########\n')
#         f.write(f'Accuracy:  {acc}\n')
#         f.write(f'PosPred: {positive_pred}\n')
#         f.write(f'FalseNeg: {false_neg}\n')
#         f.write(f'F1score: {f1score}\n')
        
        
