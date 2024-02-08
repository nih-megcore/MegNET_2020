#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 16:05:31 2024
@author: Amit Jaiswal, Megin Oy, Espoo, Finland  <amit.jaiswal@megin.fi> 
USAGE: 
    * This script is meant for quality check of ICA applied through MEGnet.
    * Run as: megnet_qc_plots.py --results_dir <directory holding megnet outputs>
    * Check as: megnet_qc_plots.py --help
"""
import mne
from os import path, makedirs, cpu_count
from glob import glob
from mne.utils import verbose
import matplotlib.pyplot as plt
plt.rcParams.setdefault
plt.rcParams["axes.xmargin"] = 0
print(__doc__)

@verbose
def plot_all(results_dir=None, ica_file=None, data_file=None, apply_filter=False, block=False, verbose=None):    
    if results_dir is not None or None in [ica_file, data_file]:
        print('Seems like you look for MEGnet quality check; searching for data_file and ica_file...')
        
        results_dir = path.abspath(results_dir)
        data_file = glob(f'{results_dir}/*_250srate_meg.fif')
        if len(data_file)==1:   data_file = data_file[0] 
        else: raise ValueError(f'Single data_file was expected but {len(data_file)} files were found '
                               f'in {results_dir}, i.e., {data_file}')
        
        ica_file = glob(f'{results_dir}/*_0-ica_applied.fif')
        ica_file = ica_file[0] if len(ica_file)==1 else \
            ValueError(f'Single ica_file was expected but {len(ica_file)} files were found in {results_dir}, '
                       f'i.e., {ica_file}')
        print(f'data_file: {data_file}')
        print(f'ica_file : {ica_file}')
        
    ica = mne.preprocessing.read_ica(ica_file)
    print(f'\n{ica}\n')
    raw = mne.io.read_raw_fif(data_file, allow_maxshield=True, preload=True)
    if not apply_filter and raw.info['lowpass']>100:
        mne.utils.warn(f"\n\nWARNING: Lowpass is {raw.info['lowpass']}, bandpass should be applied. Use --apply_filter.")
    if apply_filter:
        print(f"Data lowpass = {raw.info['lowpass']}; applying filter...")
        raw.notch_filter(raw.info['line_freq'], picks=['meg', 'eeg', 'eog', 'ecg'], filter_length='auto', 
                         notch_widths=2, trans_bandwidth=1.0, n_jobs=cpu_count(), verbose=None)
        raw.filter(1, 98, picks=['meg', 'eeg', 'eog', 'ecg'], filter_length='auto', l_trans_bandwidth='auto', 
                   h_trans_bandwidth='auto', n_jobs=cpu_count(), method='fir', phase='zero', fir_window='hamming', 
                   fir_design='firwin', verbose=None)
    
    MEGnetExtDir = f"{path.dirname(ica_file)}/MEGnetExtPlots/{data_file.split('/')[-1][:-4]}/"
    save_cfg = dict(dpi='figure', format='png')
    makedirs(f'{MEGnetExtDir}', exist_ok=True)
    fid = open(f'{MEGnetExtDir}/Explained_variance_ratio.csv', 'w')
    exp_var = ica.get_explained_variance_ratio(raw)
    sen_types = set(raw.get_channel_types())
    if 'grad' not in sen_types:
        fid.writelines("data_file, \tmag\n")
        fid.writelines(f"{data_file.split('/')[-1]}, \t{exp_var['mag']}")
    elif 'mag' not in sen_types:
        fid.writelines("data_file, \tgrad\n")
        fid.writelines(f"{data_file.split('/')[-1]}, \t{exp_var['grad']}")
    else:
        fid.writelines("data_file, \tgrad, \tmag\n")
        fid.writelines(f"{data_file.split('/')[-1]}, \t{exp_var['grad']}, \t{exp_var['mag']}")
    fid.close()
    
    comp_plot = ica.plot_components(title=f'Removed ICs indices: {ica.exclude}', show=False)
    plt.show(block=False);   plt.pause(0.001)
    comp_plot.savefig(f'{MEGnetExtDir}/all_comp_plot.png', **save_cfg)
    src_plot = ica.plot_sources(raw, picks=range(ica.n_components_), show=False, block=False)
    plt.show(block=block);   plt.pause(0.001)
    src_plot.savefig(f'{MEGnetExtDir}/all_comp_time_series_plot.png', **save_cfg)
    
    figs = ica.plot_properties(raw.copy().pick(['meg', 'eeg', 'eog', 'ecg']),
                               picks=range(ica.n_components_), show=False)
    for ii, fig in enumerate(figs):
        fig.savefig(f'{MEGnetExtDir}/properties_plot_IC{str(ii).zfill(3)}.png', **save_cfg)
        plt.close()
        
    for check in ['eog', 'ecg']:
        try:
            idxs, scores = eval(f'ica.find_bads_{check}(raw)')
            score_plot = ica.plot_scores(scores, show=False, title=f'ICA component scores ({check.upper()})',
                                         figsize=(min(20, fig.get_figwidth()*3), min(6,fig.get_figheight()*3)))
            score_plot.savefig(f'{MEGnetExtDir}/score_plot_{check.upper()}.png', **save_cfg)
            plt.close()
            del score_plot
        except Exception as exerr:
            print(exerr)
    
    ica.plot_overlay(raw, show=False)
    plt.show(block=False);   plt.pause(0.001)
    for ch_type in list(set(raw.copy().pick('meg').get_channel_types())):
        fig = plt.gcf()
        fig.set_figwidth(min(20, fig.get_figwidth()*3))
        fig.set_figheight(min(6,fig.get_figheight()*3))
        fig.savefig(f'{MEGnetExtDir}/overlay_plot_{ch_type}.png', **save_cfg)
        plt.close()
        del fig
    print(f'\nCheck plots in directory:\n{MEGnetExtDir}')

if __name__ == '__main__':
    plt.ioff()
    import argparse
    parser = argparse.ArgumentParser(description='An extension tool to plot ica components for post-operation quality check.')
    parser.add_argument('-dir',    '--results_dir',  default=None, type=str, help='Path where ICA results are saved.')
    parser.add_argument('-ica',    '--ica_file',     default=None, type=str, help='ica file.')
    parser.add_argument('-data',   '--data_file',    default=None, type=str, help='MEG data file')
    parser.add_argument('-filter', '--apply_filter', action='store_true',    help='Apply filter, in case of raw?')
    parser.add_argument('-block',  '--block',        action='store_true',    help='Block figure?')
    args = parser.parse_args()
    # print(args)
    plot_all(results_dir=args.results_dir, ica_file=args.ica_file, data_file=args.data_file, 
             apply_filter=args.apply_filter, block=args.block)
    print('...DONE.')
    
