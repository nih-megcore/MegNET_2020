#!/usr/bin/env python
# coding: utf-8


import numpy as np
import matplotlib.pyplot as plt
import mne
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from scipy import interpolate

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
    
    outlines = outlines_dict
    
    return outlines

# =============================================================================
# 
# =============================================================================

# class generate_

def _make_ica(filename):
    # read in data - for final version remove the cropping
    raw=mne.io.read_raw_fif(filename)
    raw.crop(tmax=60).pick_types(meg='mag')
    raw.load_data()
    
    # filter data and perform ICA
    n_components = 20
    filt_raw = raw.copy().filter(l_freq=1,h_freq=None)
    ica=mne.preprocessing.ICA(n_components=n_components,max_iter='auto')
    ica.fit(filt_raw)
    return ica

# # Def test
def test_:
    filename = 'data/hariri_data/sub-ON97504_ses-01_task-haririhammer_run-01_meg_250srate_meg.fif'
    raw = mne.io.read_raw_fif(filename).pick_types(meg='mag')
    ica = _make_ica(filename)
    mags = raw.ch_names

# =============================================================================
# Process
# =============================================================================

def create_circle_plot(raw, ica):
    mags = raw.ch_names
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
    channel_locations_2d=np.zeros([len(mags),2])
    newR=np.zeros((len(mags),))
    newR = 1 - PHI/np.pi*2
    channel_locations_2d=np.transpose(pol2cart(newR,TH))
    X=channel_locations_2d[:,0]
    Y=channel_locations_2d[:,1]
    
    # use ConvexHull to get the sensor indices around the edges, 
    # and scale their radii to a unit circle
    hull = ConvexHull(channel_locations_2d)
    Border=hull.vertices
    Dborder = 1/newR[Border]
    
    # Define an interpolation function of the TH coordinate to define a scaling factor for R
    FuncTh=np.hstack([TH[Border]-2*np.pi, TH[Border], TH[Border]+2*np.pi]) #.reshape((57,));  #<<<< 57 doesnt work - does this need to be here
    funcD=np.hstack((Dborder,Dborder,Dborder))
    finterp = interpolate.interp1d(FuncTh,funcD);
    D = finterp(TH)
    
    # Apply the scaling to every radii coordinate and transform back to Cartesian coordinates
    newerR=np.zeros((len(mags),))
    for i in np.arange(0,len(mags)):
        newerR[i] = min(newR[i]*D[i],1)
    [Xnew,Ynew]=pol2cart(newerR,TH)
    pos_new=np.transpose(np.vstack((Xnew,Ynew)))
    
    # create a circular outline without nose and ears, and get coordinates
    outlines_new = make_head_outlines_new(np.array([0,0,0,1]),pos_new,'head',(0,0))
    outline_coords=np.array(outlines_new['head'])
    
    # masking the outline only works on some matplotlib backends, so I've commented it out
    for comp in np.arange(0,n_components,1):
        data = np.dot(ica.mixing_matrix_[:,comp].T,ica.pca_components_[:ica.n_components_])
        mne.viz.plot_topomap(data,pos_new,sensors=False,outlines=outlines_new,extrapolate='head',sphere=[0,0,0,1],contours=0,res=128)
        #plt.plot(outline_coords[0,:],outline_coords[1,:],'white')
    



