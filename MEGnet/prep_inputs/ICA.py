#!/usr/bin/env python
# coding: utf-8

# @author: Allison Nugent
# @author: Jeff Stout

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os, os.path as op
from pathlib import PosixPath
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.preprocessing import ICA
import scipy as sp
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from scipy import interpolate
from scipy.io import savemat
from scipy.stats import zscore

from matplotlib.backends.backend_agg import FigureCanvas
from matplotlib.pyplot import show
import matplotlib
matplotlib.use('agg')

import logging
import unittest.mock
from mne.defaults import _INTERPOLATION_DEFAULT, _EXTRAPOLATE_DEFAULT, _BORDER_DEFAULT, _handle_default
from mne.utils import logger
import mne.viz.topomap
import mne.viz
from mne.viz.topomap import _check_extrapolate, _make_head_outlines, _prepare_topomap, _setup_interp, _get_patch, _draw_outlines
from mne.viz.utils import _setup_vmin_vmax, _get_cmap, plt_show
from scipy.io import savemat
import PIL.Image
import MEGnet
from MEGnet.megnet_utilities import fPredictChunkAndVoting_parrallel
import functools

from mne.io.ctf.ctf import RawCTF
from mne.io.kit.kit import RawKIT
from mne.io.bti.bti import RawBTi
from mne.io.fiff.raw import Raw
raw_typelist = [RawCTF, RawKIT, RawBTi, Raw]


# =============================================================================
# Helper Functions
# =============================================================================

# function to transform Cartesian coordinates to spherical coordinates
# theta = azimuth
# phi = elevation

def cart2sph(x, y, z):
    xy = np.sqrt(x*x + y*y)
    r = np.sqrt(x*x + y*y + z*z)
    theta = np.arctan2(y,x)
    phi = np.arctan2(z,xy)
    return r, theta, phi

# function to transform 2d polar coordinates to Cartesian
def pol2cart(rho,phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x,y

# function to transform 2d Cartesian coodinates to polar coordinates
def cart2pol(x,y):
    r = np.sqrt(x*x + y*y)
    theta = np.arctan2(y,x)
    return r,theta
# re-write of the MNE python function make_head_outlines() without the nose and ears, with expansion 
# of the outline to 1.01 

def make_head_outlines_new(sphere, pos, outlines, clip_origin):
    """Check or create outlines for topoplot."""
    assert isinstance(sphere, np.ndarray)
    x, y, _, radius = sphere
    del sphere

    ll = np.linspace(0, 2 * np.pi, 101)
    head_x = np.cos(ll) * radius*1.01 + x
    head_y = np.sin(ll) * radius*1.01 + y
    dx = np.exp(np.arccos(np.deg2rad(12)) * 1j)
    dx, dy = dx.real, dx.imag
    
    outlines_dict = dict(head=(head_x, head_y))
    
    # Make the figure encompass slightly more than all points
    mask_scale = 1.
    # We probably want to ensure it always contains our most
    # extremely positioned channels, so we do:
    mask_scale = max(
            mask_scale, np.linalg.norm(pos, axis=1).max() * 1.01 / radius)
    
    outlines_dict['mask_pos'] = (mask_scale * head_x, mask_scale * head_y)
    clip_radius = radius * mask_scale
    outlines_dict['clip_radius'] = (clip_radius,) * 2
    outlines_dict['clip_origin'] = clip_origin      
    
    return outlines_dict

def _mod_plot_topomap(
    data,
    pos,
    axes,
    *,
    ch_type="eeg",
    sensors=True,
    names=None,
    mask=None,
    mask_params=None,
    contours=6,
    outlines="head",
    sphere=None,
    image_interp=_INTERPOLATION_DEFAULT,
    extrapolate=_EXTRAPOLATE_DEFAULT,
    border=_BORDER_DEFAULT,
    res=64,
    cmap=None,
    vmin=None,
    vmax=None,
    cnorm=None,
    show=True,
    onselect=None,
):
    from matplotlib.colors import Normalize
    from matplotlib.widgets import RectangleSelector

    data = np.asarray(data)
    logger.debug(f"Plotting topomap for {ch_type} data shape {data.shape}")

    if isinstance(pos, mne.Info):  # infer pos from Info object
        picks = _pick_data_channels(pos, exclude=())  # pick only data channels
        pos = pick_info(pos, picks)

        # check if there is only 1 channel type, and n_chans matches the data
        ch_type = _get_channel_types(pos, unique=True)
        info_help = (
            "Pick Info with e.g. mne.pick_info and "
            "mne.io.pick.channel_indices_by_type."
        )
        if len(ch_type) > 1:
            raise ValueError("Multiple channel types in Info structure. " + info_help)
        elif len(pos["chs"]) != data.shape[0]:
            raise ValueError(
                "Number of channels in the Info object (%s) and "
                "the data array (%s) do not match. " % (len(pos["chs"]), data.shape[0])
                + info_help
            )
        else:
            ch_type = ch_type.pop()

        if any(type_ in ch_type for type_ in ("planar", "grad")):
            # deal with grad pairs
            picks = _pair_grad_sensors(pos, topomap_coords=False)
            pos = _find_topomap_coords(pos, picks=picks[::2], sphere=sphere)
            data, _ = _merge_ch_data(data[picks], ch_type, [])
            data = data.reshape(-1)
        else:
            picks = list(range(data.shape[0]))
            pos = _find_topomap_coords(pos, picks=picks, sphere=sphere)

    extrapolate = _check_extrapolate(extrapolate, ch_type)
    if data.ndim > 1:
        raise ValueError(
            "Data needs to be array of shape (n_sensors,); got "
            "shape %s." % str(data.shape)
        )

    # Give a helpful error message for common mistakes regarding the position
    # matrix.
    pos_help = (
        "Electrode positions should be specified as a 2D array with "
        "shape (n_channels, 2). Each row in this matrix contains the "
        "(x, y) position of an electrode."
    )
    if pos.ndim != 2:
        error = (
            "{ndim}D array supplied as electrode positions, where a 2D "
            "array was expected"
        ).format(ndim=pos.ndim)
        raise ValueError(error + " " + pos_help)
    elif pos.shape[1] == 3:
        error = (
            "The supplied electrode positions matrix contains 3 columns. "
            "Are you trying to specify XYZ coordinates? Perhaps the "
            "mne.channels.create_eeg_layout function is useful for you."
        )
        raise ValueError(error + " " + pos_help)
    # No error is raised in case of pos.shape[1] == 4. In this case, it is
    # assumed the position matrix contains both (x, y) and (width, height)
    # values, such as Layout.pos.
    elif pos.shape[1] == 1 or pos.shape[1] > 4:
        raise ValueError(pos_help)
    pos = pos[:, :2]

    if len(data) != len(pos):
        raise ValueError(
            "Data and pos need to be of same length. Got data of "
            "length %s, pos of length %s" % (len(data), len(pos))
        )

    norm = min(data) >= 0
    vmin, vmax = _setup_vmin_vmax(data, vmin, vmax, norm)
    if cmap is None:
        cmap = "Reds" if norm else "RdBu_r"
    cmap = _get_cmap(cmap)

    outlines = _make_head_outlines(sphere, pos, outlines, (0.0, 0.0))
    assert isinstance(outlines, dict)

    _prepare_topomap(pos, axes)

    mask_params = _handle_default("mask_params", mask_params)

    # find mask limits and setup interpolation
    extent, Xi, Yi, interp = _setup_interp(
        pos, res, image_interp, extrapolate, outlines, border
    )
    interp.set_values(data)
    Zi = interp.set_locations(Xi, Yi)()

    # plot outline
    patch_ = _get_patch(outlines, extrapolate, interp, axes)

    # get colormap normalization
    if cnorm is None:
        cnorm = Normalize(vmin=vmin, vmax=vmax)

    # plot interpolated map
    if image_interp == "nearest":  # plot over with Voronoi, more accurate
        im = _voronoi_topomap(
            data,
            pos=pos,
            outlines=outlines,
            ax=axes,
            cmap=cmap,
            norm=cnorm,
            extent=extent,
            res=res,
        )
    else:
        im = axes.imshow(
            Zi,
            cmap=cmap,
            origin="lower",
            aspect="equal",
            extent=extent,
            interpolation="bilinear",
            norm=cnorm,
            alpha=0.9
        )

    # gh-1432 had a workaround for no contours here, but we'll remove it
    # because mpl has probably fixed it
    linewidth = mask_params["markeredgewidth"]
    cont = True
    if isinstance(contours, (np.ndarray, list)):
        pass
    elif contours == 0 or ((Zi == Zi[0, 0]) | np.isnan(Zi)).all():
        cont = None  # can't make contours for constant-valued functions
    if cont:
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("ignore")
            cont = axes.contour(
                Xi, Yi, Zi, contours, colors="w", linewidths=linewidth / 2.0, negative_linestyles='solid', alpha=0.5, antialiased=True
            )
    if patch_ is not None:
        im.set_clip_path(patch_)
        if cont is not None:
            for col in cont.collections:
                col.set_clip_path(patch_)

    pos_x, pos_y = pos.T
    mask = mask.astype(bool, copy=False) if mask is not None else None
    if sensors is not False and mask is None:
        _topomap_plot_sensors(pos_x, pos_y, sensors=sensors, ax=axes)
    elif sensors and mask is not None:
        idx = np.where(mask)[0]
        axes.plot(pos_x[idx], pos_y[idx], **mask_params)
        idx = np.where(~mask)[0]
        _topomap_plot_sensors(pos_x[idx], pos_y[idx], sensors=sensors, ax=axes)
    elif not sensors and mask is not None:
        idx = np.where(mask)[0]
        axes.plot(pos_x[idx], pos_y[idx], **mask_params)

    if isinstance(outlines, dict):
        _draw_outlines(axes, outlines)

    if names is not None:
        show_idx = np.arange(len(names)) if mask is None else np.where(mask)[0]
        for ii, (_pos, _name) in enumerate(zip(pos, names)):
            if ii not in show_idx:
                continue
            axes.text(
                _pos[0],
                _pos[1],
                _name,
                horizontalalignment="center",
                verticalalignment="center",
                size="x-small",
            )

    if not axes.figure.get_constrained_layout():
        axes.figure.subplots_adjust(top=0.95)

    if onselect is not None:
        lim = axes.dataLim
        x0, y0, width, height = lim.x0, lim.y0, lim.width, lim.height
        axes.RS = RectangleSelector(axes, onselect=onselect)
        axes.set(xlim=[x0, x0 + width], ylim=[y0, y0 + height])
    plt_show(show)
    return im, cont, interp

# =============================================================================
# >>> Monkey patch the new topography output using the code above <<<
# Required to match the brainstorm style outputs used for training orig MEGnet
# =============================================================================
mne.viz.topomap._plot_topomap=_mod_plot_topomap
# =============================================================================

def read_raw(filename, do_assess_bads=False):
    '''
    Use the appropriate MNE io reader for the MEG type
    For CTF/.ds datasets, gradient compensation will be checked and applied if
    needed.

    Parameters
    ----------
    filename : Path or PathStr
        Path to file

    Returns
    -------
    Raw MNE instance

    '''
    #Case of BTI in folder
    if (op.isdir(str(filename))) & (not str(filename).endswith('.ds')):
        return mne.io.read_raw_bti(filename, preload=True, 
                                  head_shape_fname=None)
    
    ext = os.path.splitext(filename)[-1]
    if ext == '.fif':
        raw = mne.io.read_raw_fif(filename, preload=True, allow_maxshield=True)
        if do_assess_bads==True:
            _bads=assess_bads(filename)
            raw.info['bads'] = _bads['noisy'] + _bads['flat']
    elif ext == '.ds':
        raw = mne.io.read_raw_ctf(filename, preload=True, 
                                  system_clock='ignore', clean_names=True)
        if raw.compensation_grade != 3:
            raw.apply_gradient_compensation(3)
    #XXX Hack -- figure out the correct way to identify 4D/BTI data
    elif (filename[-4:]=='rfDC') | ('c,rfhp' in filename[-14:]):
        raw = mne.io.read_raw_bti(filename, preload=True, 
                                  head_shape_fname=None)
    #XXX Hack - Confirm KIT assignment - sqd or con file
    elif ext == '.sqd':
        raw = mne.io.read_raw_kit(filename, preload=True)
    return raw

def assess_bads(raw_fname, is_eroom=False): # assess MEG data for bad channels
    '''Code sampled from MNE python website
    https://mne.tools/dev/auto_tutorials/preprocessing/\
        plot_60_maxwell_filtering_sss.html'''
    from mne.preprocessing import find_bad_channels_maxwell
    # load data with load_data to ensure correct function is chosen
    raw = read_raw(raw_fname, do_assess_bads=False)    
    #if raw.times[-1] > 60.0:
    #    raw.crop(tmax=60)    
    raw.info['bads'] = []
    raw_check = raw.copy()
    
    vendor = mne.channels.channels._get_meg_system(raw.info)
    
    if vendor == '306m' or vendor == '122m':
        
        if is_eroom==False:
            auto_noisy_chs, auto_flat_chs, auto_scores = find_bad_channels_maxwell(
                raw_check, cross_talk=None, calibration=None,
                return_scores=True, verbose=True)
        else:
            auto_noisy_chs, auto_flat_chs, auto_scores = find_bad_channels_maxwell(
                raw_check, cross_talk=None, calibration=None,
                return_scores=True, verbose=True, coord_frame="meg")
            
        # find_bad_channels_maxwell is actually pretty bad at finding flat channels - 
        # it uses a much too stringent threshold. So, we need some supplementary code
        # This is extra complicated for Elekta/MEGIN, because there are both mags and 
        # grads, which will be on a different scale
            
        mags = mne.pick_types(raw_check.info, meg='mag')
        grads = mne.pick_types(raw_check.info, meg='grad')
        # get the standard deviation for each channel, and the trimmed mean of the stds
        # have to do this separately for mags and grads
        stdraw_mags = np.std(raw_check._data[mags,:],axis=1)
        stdraw_grads = np.std(raw_check._data[grads,:],axis=1)    
        stdraw_trimmedmean_mags = sp.stats.trim_mean(stdraw_mags,0.1)
        stdraw_trimmedmean_grads = sp.stats.trim_mean(stdraw_grads,0.1)
        # we can't use the same threshold here, because grads have a much greater 
        # variance in the variances 
        flat_mags = np.where(stdraw_mags < stdraw_trimmedmean_mags/100)[0]
        flat_grads = np.where(stdraw_grads < stdraw_trimmedmean_grads/1000)[0]
        # need to use list comprehensions
        flat_idx_mags = [flat_mags[i] for i in flat_mags.tolist()]
        flat_idx_grads = [flat_grads[i] for i in flat_grads.tolist()]
        flats = []
        for flat in flat_idx_mags:
            flats.append(raw_check.info['ch_names'][mags[flat_idx_mags]])
        for flat in flat_idx_grads:
            flats.append(raw_check.info['ch_names'][grads[flat_idx_grads]])
        
    # ignore references and use 'meg' coordinate frame for CTF and KIT
    
    if vendor == 'CTF_275':
        raw_check.apply_gradient_compensation(0)
        auto_noisy_chs, auto_flat_chs, auto_scores = find_bad_channels_maxwell(
            raw_check, cross_talk=None, calibration=None, coord_frame='meg',
            return_scores=True, verbose=True, ignore_ref=True)
        
        # again, finding flat/bad channels is not great, so we add another algorithm
        # since other vendors don't mix grads and mags, we only need to do this for
        # a single channel type
        
        megs = mne.pick_types(raw_check.info, meg=True)
        # get the standard deviation for each channel, and the trimmed mean of the stds
        stdraw_megs = np.std(raw_check._data[megs,:],axis=1)
        stdraw_trimmedmean_megs = sp.stats.trim_mean(stdraw_megs,0.1)
        flat_megs = np.where(stdraw_megs < stdraw_trimmedmean_megs/100)[0]
        # need to use list comprehensions
        flat_idx_megs = [flat_megs[i] for i in flat_megs.tolist()]
        flats = []
        for flat in flat_idx_megs:
            flats.append(raw_check.info['ch_names'][megs[flat_idx_mags]]) 
    
    else: 
        
        auto_noisy_chs, auto_flat_chs, auto_scores = find_bad_channels_maxwell(
            raw_check, cross_talk=None, calibration=None, coord_frame='meg',
            return_scores=True, verbose=True, ignore_ref=True)
    
        # again, finding flat/bad channels is not great, so we add another algorithm
        # since other vendors don't mix grads and mags, we only need to do this for
        # a single channel type
    
        megs = mne.pick_types(raw_check.info, meg=True)
        # get the standard deviation for each channel, and the trimmed mean of the stds
        stdraw_megs = np.std(raw_check._data[megs,:],axis=1)
        stdraw_trimmedmean_megs = sp.stats.trim_mean(stdraw_megs,0.1)
        flat_megs = np.where(stdraw_megs < stdraw_trimmedmean_megs/100)[0]
        # need to use list comprehensions
        flat_idx_megs = [flat_megs[i] for i in flat_megs.tolist()]
        flats = []
        for flat in flat_idx_megs:
            flats.append(raw_check.info['ch_names'][megs[flat_idx_mags]])    
    
    auto_flat_chs = auto_flat_chs + flats
    auto_flat_chs = list(set(auto_flat_chs))
            
    return {'noisy':auto_noisy_chs, 'flat':auto_flat_chs}         

def raw_preprocess(raw, mains_freq=None):
    '''
    Preprocess data with 250Hz resampling, notch_filter, and 1-100Hz bp
    Returns instance of mne raw
    '''
    resample_freq = 250
    mains_freq = int(mains_freq)
    notch_freqs = np.arange(mains_freq, resample_freq * 2/3, mains_freq)
    raw.notch_filter(notch_freqs) 
    raw.resample(resample_freq)
    raw.filter(1.0, 100)
    return raw

def thresh_get_good_segments(raw):
    magthresh = 5000e-15
    gradthresh = 5000e-13
    flatmagthresh = 10e-15
    flatgradthresh = 10e-13
    evts = mne.make_fixed_length_events(raw, duration=5.0)        
    chtypes=raw.get_channel_types()
    if 'grad' in chtypes:
        if 'mag' in chtypes: 
            reject_dict = dict(mag=magthresh, grad=gradthresh)
            flat_dict = dict(mag=flatmagthresh, grad=flatgradthresh)
        else:
            reject_dict = dict(grad=gradthresh)
            flat_dict = dict(grad=flatgradthresh)
    else:
        reject_dict = dict(mag=magthresh)
        flat_dict = dict(mag=flatmagthresh)
    epochs = mne.Epochs(raw, evts, reject=reject_dict, flat=flat_dict, preload=True, baseline=None)
    return epochs

def z_get_good_segments(epochs, std_thresh=6):
    '''Identify bad channels using standard deviation'''
    epochs = epochs.copy()
    z = zscore(np.std(epochs._data, axis=2), axis=0)
    bad_epochs = np.where(z>std_thresh)[0]
    epochs.drop(indices=bad_epochs)
    return epochs
           
def calc_ica(raw, file_base=None, save=False, results_dir=None, seedval=0):
    '''Straightforward MNE ICA with MEGnet article specifications:
        infomax, 20 components'''
    epochs = thresh_get_good_segments(raw)
    epochs = z_get_good_segments(epochs)
    ica = ICA(n_components=20, random_state=seedval, method='infomax')
    ica.fit(epochs)
    if save==True:
        out_filename = file_base + '_{}-ica.fif'.format(str(seedval))
        out_filename = os.path.join(results_dir, out_filename)
        ica.save(out_filename, overwrite=True)
    return ica

# =============================================================================
# Neighborhood Correlation
# =============================================================================
from sklearn.neighbors import NearestNeighbors
def get_sensor_locs(raw):
    '''Return sensor coordinates'''
    locs = np.array([i['loc'][0:3] for i in raw.info['chs']])
    return locs

def get_neighbors(raw=None, n_neighbors=6):
    '''Enter the MNE raw object
    Returns neighborhood index matrix with the first column being the index
    of the input channel - and the rest of the columns being the neighbor 
    indices.  n_neighbors defines the number of neighbors found'''
    if raw!=None:
        locs=get_sensor_locs(raw)
    n_neighbors+=1  #Add 1 - because input chan is one of hte "neighbors" 
        
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree')
    nbrs.fit(locs)
    distances, neighbor_mat = nbrs.kneighbors(locs) 
    return distances, neighbor_mat
    
# def return_index(val, idx):
#     return val[idx]

def neighborhood_corr(raw, n_neighbors=6):
    dists, neighbor_mat = get_neighbors(raw) 
    corr_vec=np.zeros(neighbor_mat.shape[0])
    for idx,row in enumerate(neighbor_mat):
        tmp = (np.corrcoef(raw._data[row])[0,1:] / dists[idx][1:]) * dists[idx][1:].mean()
        corr_vec[idx] = np.abs(tmp.mean())

# =============================================================================
#     
# =============================================================================

def sensor_pos2circle(raw, ica):
    '''
    Project the sensor positions to a unit circle and return positions
    Currently works with MNE chan_type == mag  (includes CTF ax gradiometers)

    Parameters
    ----------
    raw : mne.io.{fiff,ds}.raw.Raw
        Mne format dataset.  
    ica : mne.preprocessing.ica.ICA
        MNE ICA instance

    Returns
    -------
    pos_new : numpy.ndarray 
        Position of channels projected to the unit circle. (#chans X 2)

    '''
    num_chans = len(raw.ch_names)
    # extract magnetometer positions
    data_picks, pos, merge_channels, names, ch_type, sphere, clip_origin = \
        mne.viz.topomap._prepare_topomap_plot(ica, 'mag')
    
    #Extract channel locations
    # 'loc' has 12 elements, the location plus a 3x3 orientation matrix 
    tmp_ = [i['loc'][0:3] for i in raw.info['chs']]
    channel_locations3d = np.stack(tmp_)
    
    tmp_ = np.array([cart2sph(*i) for i in channel_locations3d])
    channel_locations_3d_spherical = tmp_ #np.transpose(tmp_) 
    
    TH=channel_locations_3d_spherical[:,1]
    PHI=channel_locations_3d_spherical[:,2]
    
    # project the spherical locations to a plane
    # this calculates a new R for each coordinate, based on PHI
    # then transform the projection to Cartesian coordinates
    channel_locations_2d=np.zeros([num_chans,2])
    newR=np.zeros((num_chans,))
    newR = 1 - PHI/np.pi*2
    channel_locations_2d=np.transpose(pol2cart(newR,TH))
    
    # use ConvexHull to get the sensor indices around the edges, 
    # and scale their radii to a unit circle
    hull = ConvexHull(channel_locations_2d)
    Border=hull.vertices
    Dborder = 1/newR[Border]
    
    # Define an interpolation function of the TH coordinate to define a scaling 
    # factor for R
    FuncTh=np.hstack([TH[Border]-2*np.pi, TH[Border], TH[Border]+2*np.pi]) 
    funcD=np.hstack((Dborder,Dborder,Dborder))
    finterp = interpolate.interp1d(FuncTh,funcD);
    D = finterp(TH)
    
    # Apply the scaling to every radii coordinate and transform back to 
    # Cartesian coordinates
    newerR=np.zeros((num_chans,))
    for i in np.arange(0,num_chans):
        newerR[i] = min(newR[i]*D[i],1)
    [Xnew,Ynew]=pol2cart(newerR,TH)
    pos_new=np.transpose(np.vstack((Xnew,Ynew)))
    return pos_new



def circle_plot(circle_pos=None, data=None, out_fname=None):
    '''Generate the plot and save'''
    # create a circular outline without nose and ears, and get coordinates
    outlines_new = make_head_outlines_new(np.array([0,0,0,1]),
                                          circle_pos,
                                          'head',
                                          (0,0))
    fig = plt.figure(figsize=(1.5, 1.8), dpi=100)
    ax = fig.add_subplot(111)
    ax.set_facecolor('k')
    
    mnefig, contour = mne.viz.plot_topomap(data,circle_pos,
                                sensors=False,
                                outlines=outlines_new,
                                extrapolate='head',
                                sphere=[0,0,0,1.0],
                                contours=9,res=120,
                                show=False,
                                axes=ax, 
                                )
    contour.colors=['white']  #This is not working currently 
    mnefig.set_cmap(plt.get_cmap('bwr'))
    # plt.show()
    mnefig.figure.savefig(out_fname)
    plt.close(fig)
    
    mat_fname = os.path.splitext(out_fname)[0]+'.mat'
    
    matrix_out = np.frombuffer(mnefig.figure.canvas.tostring_rgb(), dtype=np.uint8)
    matrix_out = matrix_out.reshape(mnefig.figure.canvas.get_width_height()[::-1] + (3,))
    savemat(mat_fname, 
            {'array':matrix_out})
    del mnefig
    #return matrix_out
    

def main(filename, outbasename=None, mains_freq=60.0, 
             save_preproc=False, save_ica=False, seedval=0,
             results_dir=None, filename_raw=None, do_assess_bads=False,
             bad_channels=[]):
    '''
        Perform all of the steps to preprocess the ica maps:
        Read raw data
        Filter / notch filter / resample (250Hz) 
        Calculate 20 component ICA using infomax
        Warp sensors to circle plot
        Save ICA to output dir
        Save ICA topoplots to output dir
        
        Parameters
        ----------
        
        filename : str or Raw MNE data object
            Path to file
        filename_raw : str
            Required for MEGIN datasets
            Path to file
        outbasename : str
            Required for 4D/BTI datasets
            If none defaults to basename(filename)
        mains_freq : float
            Line frequency 50 or 60 Hz
        save_preproc : Bool 
            Save the preprocessed data
        save_ica : Bool 
            Save the ica output 
        seedval : Int
            Set the numpy random seed
        results_dir : str / path
            Path to output directory
        do_assess_bads : Bool
            Assess bad channels if not already done
            
    '''
    if (type(filename) == str) | (type(filename) == PosixPath):
        raw = read_raw(filename)
    elif type(filename) in raw_typelist:
        raw = deepcopy(filename)
    else:
        raise BaseException('Could not interpret input variable "filename"')
        
    if len(bad_channels) > 0:
        print('dropping bad channels\n')
        print(bad_channels)
        raw.drop_channels(bad_channels)
    raw = raw_preprocess(raw, mains_freq)

    if filename_raw is not None:
        tmp_raw = read_raw(filename_raw, do_assess_bads=do_assess_bads)
        tmp_raw = raw_preprocess(tmp_raw, mains_freq)    
        raw.info['bads'] = tmp_raw.info['bads']
        del tmp_raw
    
    #Set output names
    if outbasename != None:
        file_base = outbasename #Necessary for 4D datasets
    else:
        file_base = os.path.basename(filename)
        file_base = os.path.splitext(file_base)[0]
    
    results_dir = os.path.join(results_dir, file_base)
    if not os.path.exists(results_dir): os.mkdir(results_dir)
    
    if save_preproc==True:
        out_fname = os.path.join(results_dir, file_base+'_250srate_meg.fif')
        raw.save(out_fname, overwrite=True) #Save with EEG
    
    # pick only the MEG channels, and then grab the indices of the channels we want to use 
    raw.pick_types(meg=True, eeg=False, ref_meg=False)
    mag_idxs = mne.pick_types(raw.info, ref_meg=False, meg='mag')

    ica = calc_ica(raw, file_base=file_base, results_dir=results_dir,
                   save=save_ica, seedval=seedval)
        
    # generate the sensor positions on the circle
    circle_pos = sensor_pos2circle(raw, ica)
    
    for comp in range(0,20):
      # get the ICA component data
      data=np.dot(ica.mixing_matrix_[:,comp].T, ica.pca_components_[:ica.n_components_])
      data = data[mag_idxs]
    
      '''Generate the plot and save'''
      # create a circular outline without nose and ears, and get coordinates
      outlines_new = make_head_outlines_new(np.array([0,0,0,1]),
                                              circle_pos,
                                              'head',
                                              (0,0))
      # set up the figure canvas
      fig = plt.figure(figsize=(1.3, 1.3), dpi=100, facecolor='black')
      canvas=FigureCanvas(fig)
      ax = fig.add_subplot(111)
      
      # plot the figure using the monkey patched mne.viz.plot_topomap
      mnefig, contour = mne.viz.plot_topomap(data,
                                    circle_pos[mag_idxs],
                                    sensors=False,
                                    outlines=outlines_new,
                                    extrapolate='head',
                                    sphere=[0,0,0,1.],
                                    contours=10,res=120,
                                    show=True,
                                    axes=ax, 
                                    cmap='bwr'
                                    )
      
      print(comp)
      # plt.show()
    
      # So matplotlib did strange things to the canvas and background image size when I tried to grab
      # the image from the canvas and convert directly to RGB. This ridiculous (but functional) hack is to 
      # write the image out as a png then reopen it to convert to RGB. 
      outpng = f'{results_dir}/component{str(int(comp)+1)}.png'
      mnefig.figure.savefig(outpng,dpi=120,bbox_inches='tight',pad_inches=0)
      rgba_image=PIL.Image.open(outpng)
      rgb_image=rgba_image.convert('RGB')
    
      # save the RGB image as a .mat file
      mat_fname = f'{results_dir}/component{str(int(comp)+1)}.mat'
      savemat(mat_fname, {'array':np.array(rgb_image)})
      del mnefig

      # Save ICA timeseries as input for classification
      # Currently inputs to classification are matlab arrays
      ica_ts = ica.get_sources(raw)._data.T
      outfname = f'{results_dir}/ICATimeSeries.mat' #'{file_base}-ica-ts.mat'
      savemat(outfname, {'arrICATimeSeries':ica_ts})
    
def classify_ica(results_dir=None, outbasename=None, filename=None):
    '''
    Run the ICA timeseries and spatial maps generated during the main processing
    through the MEGNET tensorflow model (using the CPU)

    Parameters
    ----------
    results_dir : str, optional
        Path to the output directory from the main processing
        Contains the spatial maps and time series

    Returns
    -------
    dict
        dictionary with keys: (classes, bads_idx)

    '''
    from scipy.io import loadmat
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
    from tensorflow import keras
    model_path = op.join(MEGnet.__path__[0] ,  'model_v2')
    # This is set to use CPU in initial import
    kModel=keras.models.load_model(model_path)
    
    #Set output names
    if outbasename != None:
        file_base = outbasename #Necessary for 4D datasets
    else:
        file_base = os.path.basename(filename)
        file_base = os.path.splitext(file_base)[0]
    results_dir = os.path.join(results_dir, file_base)
    
    outfname = f'{results_dir}/ICATimeSeries.mat'
    arrSP_fnames = [op.join(results_dir, f'component{i}.mat') for i in range(1,21)]
    arrTS = loadmat(op.join(results_dir, 'ICATimeSeries.mat'))['arrICATimeSeries'].T
    arrSP = np.stack([loadmat(i)['array'] for i in arrSP_fnames])
    preds, probs = fPredictChunkAndVoting_parrallel(kModel, arrTS, arrSP)
    meg_rest_ica_classes = preds.argmax(axis=1)
    ica_comps_toremove = [index for index, value in enumerate(meg_rest_ica_classes) if value in [1, 2, 3]]
    return {'classes':meg_rest_ica_classes,
            'bads_idx': ica_comps_toremove}

def clean_ica(bad_comps=None, results_dir=None, outbasename=None,
              raw_dataset=None):         # Remove identified ICA components    
    print("removing ica components")
        #Set output names
    if outbasename != None:
        file_base = outbasename #Necessary for 4D datasets
    else:
        file_base = os.path.basename(filename)
        file_base = os.path.splitext(file_base)[0]
    results_dir = os.path.join(results_dir, file_base)
    
    ica_fname = op.join(results_dir, file_base +'_0-ica.fif')
    ica=mne.preprocessing.read_ica(ica_fname) 
    ica.exclude = bad_comps
    raw=load_data(raw_dataset)
    ica.apply(raw)    
    outfname = op.join(results_dir, 'ica_clean.fif')
    raw.save(outfname)
    ica.save(ica_fname.replace('_0-ica.fif', '_0-ica_applied.fif'))
    
def check_datatype(filename):               # function to determine the file format of MEG data 
    '''Check datatype based on the vendor naming convention to choose best loader'''
    if os.path.splitext(filename)[-1] == '.ds':
        return 'ctf'
    elif os.path.splitext(filename)[-1] == '.fif':
        return 'fif'
    elif os.path.splitext(filename)[-1] == '.4d' or ',' in str(filename):
        return '4d'
    elif os.path.splitext(filename)[-1] == '.sqd':
        return 'kit'
    elif os.path.splitext(filename)[-1] == 'con':
        return 'kit'
    else:
        raise ValueError('Could not detect datatype')
        
def return_dataloader(datatype):            # function to return a data loader based on file format
    '''Return the dataset loader for this dataset'''
    if datatype == 'ctf':
        return functools.partial(mne.io.read_raw_ctf, system_clock='ignore',
                                 clean_names=True)
    if datatype == 'fif':
        return functools.partial(mne.io.read_raw_fif, allow_maxshield=True)
    if datatype == '4d':
        return mne.io.read_raw_bti
    if datatype == 'kit':
        return mne.io.read_raw_kit

def load_data(filename):                    # simple function to load raw MEG data
    datatype = check_datatype(filename)
    dataloader = return_dataloader(datatype)
    raw = dataloader(filename, preload=True)
    return raw

#%%
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser() 
    # bids_args = parser.add_argument_group('BIDS')
    # bids_args.add_argument('-bids_root', help='foo help')
    # bids_args.add_argument('-bids_id', help='BIDS ID')
    # bids_args.add_argument('-run', help='Run Number')
    # bids_args.add_argument('-task', help='Task name')
    
    parser.add_argument('-filename', help='Path to MEG dataset')
    parser.add_argument('-filename_raw', help='''Required for MEGIN data.
                        Path to the non-SSS data.  Do not use this data for 
                        non-MEGIN data.
                        ''')
    parser.add_argument('-outbasename', 
                        help='''Basename for output directory.  If none is 
                        provided, the basename up to the suffix of the file 
                        will be used as the folder name inside of the results_dir.
                        NOTE: this is a required flag for 4D/BTI data - since the 
                        standard filenames are not unique''', 
                        required=False)
    parser.add_argument('-results_dir', help='Path to save the results')
    parser.add_argument('-line_freq', help='{60,50} Hz - AC electric frequency')
    # parser.add_argument('-clean_data', 
    #                     help='''Perform classification of the ICA components and
    #                     generate the ICA cleaned dataset''',
    #                     default=True)
    args = parser.parse_args()
    
    filename = args.filename
    mains_freq = float(args.line_freq)
    
    main(filename, outbasename=args.outbasename, mains_freq=mains_freq, 
             save_preproc=True, save_ica=True, seedval=0, filename_raw=args.filename_raw,
             results_dir=args.results_dir)
    
    ica_dict = classify_ica(results_dir=args.results_dir, outbasename=args.outbasename,
                            filename=filename)
    
    clean_ica(bad_comps=ica_dict['bads_idx'], results_dir=args.results_dir,
              raw_dataset=args.filename, outbasename=args.outbasename)
    
    
    
    
    
    

