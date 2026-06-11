#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 21:23:50 2023

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
from MEGnet.megnet_utilities import fGetStartTimesOverlap
from numpy.random import randint
from numpy.random import shuffle
import copy

tmp=MEGnet.__path__[0]


train_dir = op.join(MEGnet.__path__[0], 'prep_inputs','training')
pkl_topdir = op.join(train_dir, 'Inputs', 'PKL')

# =============================================================================
# Load the already saved arrays (check the new_train_model_pkl.py for this process).
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

assert tr_arrTS.shape[0]==tr_arrSP.shape[0]
assert tr_arrCL.shape[0]==tr_arrSP.shape[0]

# =============================================================================
# 
# =============================================================================
NB_EPOCH = 1 #200
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


import shutil
def save_weights_and_history(history, kModel, outdir=None):
    if not os.path.exists(outdir): os.mkdir(outdir)
    with open(f'{outdir}/history.pkl', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    kModel.save(f'{outdir}/model')
    # with open(f'{outdir}/score', 'wb') as file_sc:
    #     pickle.dump(score[idx], file_sc)
    shutil.copy(os.path.realpath(__file__), outdir)
    

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

history = kModel.fit(
    train_seq,
    epochs=NB_EPOCH,
    verbose=VERBOSE,
    validation_data=(dict(spatial_input=te_arrSP, temporal_input=te_arrTS),
                     one_hot(te_arrCL, 4)),
    class_weight=class_weights,
    callbacks=[earlystop],
)

out_folder = '/data/EnigmaMeg/MEGNET/MODELFIT'
import glob
_tmp = glob.glob(f'{out_folder}/MODEL_*')
_tmp = sorted(_tmp)
if len(_tmp) == 0:
    outdir = op.join(out_folder, 'MODEL_v1')
else:
    ver = str(int(_tmp[-1].split('_')[-1].replace('v',''))+1)
    outdir = op.join(out_folder, 'MODEL_v{ver}')
save_weights_and_history(history, kModel, outdir=outdir)



    
