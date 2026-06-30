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

pkl_dir = '/vf/users/EnigmaMeg/BIDS/ICA/ICA_megnet/PKL'
# pkl_dir = '/home/jstout/src/MegNET2022/MEGnet/prep_inputs/training/Inputs/PKL'
pkl_topdir = pkl_dir
dframe_hold = pd.read_csv(op.join(pkl_topdir, 'dframe_holdout_new.csv'))

# =============================================================================
# Run the prediction scoring
# =============================================================================

kModel_fname = op.join(MEGnet.__path__[0], 'model_v2')
kModel = keras.models.load_model(kModel_fname, compile=False)

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
        
        