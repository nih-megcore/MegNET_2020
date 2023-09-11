#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 18:51:35 2022

@author: jstout
Minimal testing - needs to be improved with abstraction of tmpdir
Also needs more comparitive testing with results
Also more granular testing of functions would be preferrable
"""

import numpy as np
import mne
from ..ICA import sensor_pos2circle
from ..ICA import read_raw, raw_preprocess, calc_ica
from pathlib import Path
import os, os.path as op
from ..ICA import main, classify_ica, clean_ica
import shutil
import scipy
import pygit2
from scipy.io import savemat, loadmat


# =============================================================================
# Setup input / output folders
# =============================================================================
raw_gitdir='/tmp/TEST_ctf_data'
gt_gitdir = '/tmp/TEST_ctf_gt' #gt for ground truth
if not op.exists(raw_gitdir):
    pygit2.clone_repository('https://github.com/nih-megcore/TEST_ctf_data.git', raw_gitdir)
if not op.exists(gt_gitdir):    
    pygit2.clone_repository('https://github.com/nih-megcore/MEGnet_testing_data.git', gt_gitdir)

ctf_filename = op.join(raw_gitdir, '20010101','ABABABAB_airpuff_20010101_001.ds')
ctf_test_gt = gt_gitdir

results_dir = '/tmp/test/results'

# =============================================================================
# SSL errors with mne downloads
# =============================================================================
# download_path = '/tmp/test'
# path = mne.datasets.sample.data_path(download_path)
# megin_filename = Path(path) / 'MEG/sample/sample_audvis_raw.fif'

tests_array_dir = os.path.dirname(os.path.abspath(__file__))

# def test_circle_plot():
#     raw = mne.io.read_raw_fif(megin_filename).crop(60.0).pick_types(meg='mag').load_data()
#     ica = raw_preprocess(raw, mains_freq=60)
#     pos_new = sensor_pos2circle(raw, ica)
#     tmp_ = np.load(os.path.join(tests_array_dir,
#                                 'circle_plot_pos_new.npy'))
#     assert np.allclose(pos_new, tmp_)
    
def test_reader():
    raw = read_raw(ctf_filename)
    assert raw.compensation_grade == 3
    assert isinstance(raw, mne.io.ctf.ctf.RawCTF)
    
#     raw = read_raw(megin_filename)
#     assert isinstance(raw, mne.io.fiff.raw.Raw)

# def test_megin():
#     filename = '/tmp/test/MNE-sample-data/MEG/sample/sample_audvis_raw.fif'
#     results_dir = '/tmp/test/results'
#     if os.path.exists(results_dir) : shutil.rmtree(results_dir)
#     os.mkdir(results_dir)
#     main(
#         filename,
#         mains_freq=60,
#         save_preproc=True,
#         save_ica=True,
#         seedval=0,
#         results_dir=results_dir
#         )
    
def test_ctf():
    '''Regression test for ctf data'''
    if os.path.exists(results_dir) : shutil.rmtree(results_dir)
    os.mkdir(results_dir)
    main(
        ctf_filename,
        mains_freq=60,
        save_preproc=True,
        save_ica=True,
        seedval=0,
        results_dir=results_dir
        )
    outbasename = op.basename(ctf_filename)[:-3]
    outdir = op.join(results_dir, outbasename)
    assert op.exists(op.join(outdir,'ICATimeSeries.mat'))
    for i in range(20):
        assert op.exists(op.join(outdir, f'component{i+1}.png'))
        assert op.exists(op.join(outdir, f'component{i+1}.mat'))
    ica_ts = scipy.io.loadmat(op.join(outdir, 'ICATimeSeries.mat'))['arrICATimeSeries']
    ica_ts_gt = scipy.io.loadmat(op.join(gt_gitdir, 'ICATimeSeries.mat'))['arrICATimeSeries']
    assert np.allclose(ica_ts, ica_ts_gt)
    
    raw = mne.io.read_raw_fif(op.join(outdir, 'ABABABAB_airpuff_20010101_001_250srate_meg.fif'))
    raw_gt = mne.io.read_raw_fif(op.join(gt_gitdir, 'ABABABAB_airpuff_20010101_001_250srate_meg.fif'))
    assert np.allclose(raw.get_data(), raw_gt.get_data())

    ica = mne.preprocessing.read_ica(op.join(outdir, 'ABABABAB_airpuff_20010101_001_0-ica.fif'))
    ica_gt = mne.preprocessing.read_ica(op.join(gt_gitdir, 'ABABABAB_airpuff_20010101_001_0-ica.fif'))                          
    assert np.allclose(ica.unmixing_matrix_ , ica_gt.unmixing_matrix_)

def test_classify_ica():
    '''Verify consistent assessments of the inputs'''
    outbasename = op.basename(ctf_filename)[:-3]
    outdir = op.join(results_dir, outbasename)
    tmp_ = op.join(results_dir, outbasename)
    
    ts_fname = op.join(tmp_, 'ICATimeSeries.mat')
    arrTS = loadmat(ts_fname)['arrICATimeSeries']
    ## Minor hack to make the time series long enough for classification
    ica_ts = np.vstack([arrTS, arrTS])
    savemat(ts_fname, {'arrICATimeSeries':ica_ts})
    # Classify the data vectors
    ica_dict = classify_ica(results_dir=results_dir, filename=ctf_filename)
    assert np.alltrue(ica_dict['classes']==[1, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    assert np.alltrue(ica_dict['bads_idx']==[0,4,5])    
    
def test_clean_ica():
    '''Confirm that the cleaned output meg matches the expected''' 
    outbasename = op.basename(ctf_filename)[:-3]
    outdir = op.join(results_dir, outbasename)
    clean_ica(bad_comps=[0,4,5], results_dir=results_dir, outbasename=outbasename,
              raw_dataset=ctf_filename)
    
    cleaned_fname = op.join(outdir, 'ica_clean.fif')
    raw_cleaned = mne.io.read_raw_fif(cleaned_fname, preload=True)    
    gt_fname = op.join(gt_gitdir, 'ica_clean.fif')
    gt_raw = mne.io.read_raw_fif( gt_fname ,preload=True)
    assert np.allclose(gt_raw.get_data(), raw_cleaned.get_data())    

    
    
    
