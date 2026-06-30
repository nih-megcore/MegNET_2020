#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 15:26:23 2026

@author: stoutjd
"""
# ============================================================================
# augmentation_generator.py
# Drop into MEGnet/prep_inputs/training/ alongside new_train_model_pkl.py
# Env: tensorflow 2.6.2, keras 2.6.0, numpy 1.19.5, scipy 1.9.1
# ============================================================================
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.ndimage import rotate
from tensorflow.keras.utils import Sequence


def _make_lowpass(cutoff_hz=125.0, sfreq=300.0, order=4):
    """Design a Butterworth low-pass. cutoff must be < Nyquist (sfreq/2)."""
    nyq = sfreq / 2.0
    assert cutoff_hz < nyq, f"cutoff {cutoff_hz} must be < Nyquist {nyq}"
    b, a = butter(order, cutoff_hz / nyq, btype='low')
    return b, a


def bandlimited_noise(shape, std, b, a, rng):
    """White noise low-pass filtered to <=cutoff. Filtered along time axis=1."""
    noise = rng.normal(0.0, std, size=shape)
    # filtfilt is zero-phase; operate along the time axis (axis=1 for [N,15000])
    return filtfilt(b, a, noise, axis=1)


# ============================================================================
# Spatial topography rotation
# ============================================================================
def _rotate_topo(sp, angle_deg, fill=255):
    """
    Rotate a spatial topography around its center pixel.

    Parameters
    ----------
    sp : np.ndarray
        Spatial map, shape (120, 120) grayscale or (120, 120, 3) RGB, uint8.
    angle_deg : float
        Counter-clockwise rotation in degrees.
    fill : int
        Value for pixels rotated in from outside the original frame.
        255 = white (matches the MEGnet background); use 0 for black.

    Returns
    -------
    np.ndarray, same shape & dtype as input.
    """
    # scipy.ndimage.rotate rotates about the center of the given axes.
    # reshape=False keeps the output 120x120 (crops the corners that spin out).
    out = rotate(sp, angle_deg, axes=(0, 1), reshape=False,
                 order=1, mode='constant', cval=fill)
    return np.clip(out, 0, 255).astype(np.uint8)


def circular_mask(shape=(120, 120), center=None, radius=None):
    """
    Boolean circular mask, True inside the circle.

    Parameters
    ----------
    shape : tuple
        (H, W) of the image.
    center : tuple or None
        (row, col) center. None -> geometric center (59.5, 59.5) for 120x120.
    radius : float or None
        Circle radius in pixels. None -> largest circle that fits.

    Returns
    -------
    np.ndarray of bool, shape == shape.
    """
    h, w = shape
    if center is None:
        center = ((h - 1) / 2.0, (w - 1) / 2.0)   # (59.5, 59.5)
    if radius is None:
        radius = min(center[0], center[1], h - 1 - center[0], w - 1 - center[1])
    yy, xx = np.ogrid[:h, :w]
    dist2 = (yy - center[0]) ** 2 + (xx - center[1]) ** 2
    return dist2 <= radius ** 2


def rotate_topo(topo_sp, angle=None, mask=None):
    """
    Rotate an RGB topography by `angle` degrees and zero everything outside
    the inscribed circle.

    Parameters
    ----------
    topo_sp : np.ndarray
        Spatial map of shape (120, 120, 3), uint8.
    angle : float
        Counter-clockwise rotation in degrees.
    mask : np.ndarray or None
        Precomputed boolean circular mask of shape (120, 120). If None, it is
        computed on the fly (slower; precompute when calling in a loop).

    Returns
    -------
    np.ndarray of shape (120, 120, 3), uint8.
    """
    if mask is None:
        mask = circular_mask((120, 120))
    out0 = _rotate_topo(topo_sp[:, :, 0], angle)
    out0[~mask] = 0
    out1 = _rotate_topo(topo_sp[:, :, 1], angle)
    out1[~mask] = 0
    out2 = _rotate_topo(topo_sp[:, :, 2], angle)
    out2[~mask] = 0
    out = np.stack([out0, out1, out2], axis=-1)
    return out


class MEGnetAugSequence(Sequence):
    """
    On-the-fly augmentation for the dual-input MEGnet model.

    Per batch:
      - adds band-limited (<=125 Hz) white noise to temporal_input
      - with p_flip probability PER SAMPLE, multiplies the time series by -1
        and simultaneously inverts the matched spatial map (255 - sp)
      - with p_rot probability PER SAMPLE, rotates the spatial map by a random
        angle in [-max_rot_deg, +max_rot_deg] using rotate_topo
    """
    def __init__(self, arrSP, arrTS, arrCL, batch_size,
                 noise_snr=0.1, p_flip=0.5, cutoff_hz=125.0,
                 sfreq=300.0, num_classes=4, shuffle=True, seed=0,
                 p_rot=0.5, max_rot_deg=10.0):
        assert arrSP.shape[0] == arrTS.shape[0] == arrCL.shape[0]
        self.arrSP = arrSP
        self.arrTS = arrTS
        self.arrCL = arrCL
        self.batch_size = batch_size
        self.noise_snr = noise_snr
        self.p_flip = p_flip
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)
        self.b, self.a = _make_lowpass(cutoff_hz, sfreq)
        self.indices = np.arange(arrTS.shape[0])
        # rotation params
        self.p_rot = p_rot
        self.max_rot_deg = max_rot_deg
        # precompute the circular mask once (reused for every rotation)
        self.mask = circular_mask((120, 120))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(self.arrTS.shape[0] / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i * self.batch_size:(i + 1) * self.batch_size]

        # copy so we never mutate the source arrays
        sp = self.arrSP[idx].copy()
        ts = self.arrTS[idx].copy()
        cl = self.arrCL[idx]

        # --- band-limited noise on the time series ---
        # scale noise by per-sample std (snr meaning matches ica_augmenter)
        ts_std = ts.std(axis=1, keepdims=True)
        ts += bandlimited_noise(ts.shape, ts_std * self.noise_snr,
                                self.b, self.a, self.rng)

        # --- coupled 50% sign flip / topography inversion ---
        flip = self.rng.random(ts.shape[0]) < self.p_flip
        ts[flip] *= -1.0
        # invert the matched spatial maps: 255 - x, clipped to valid range
        sp[flip] = sp[flip][..., ::-1]

        # --- random +/- rotation of the spatial topography ---
        # operates per-sample; rotation is spatial-only, so it does not need
        # to be coupled to any change in the time series.
        rot = self.rng.random(sp.shape[0]) < self.p_rot
        for j in np.nonzero(rot)[0]:
            angle = self.rng.uniform(-self.max_rot_deg, self.max_rot_deg)
            sp[j] = rotate_topo(sp[j], angle, mask=self.mask)

        x = {'spatial_input': sp.astype(np.float32),
             'temporal_input': ts.astype(np.float32)}
        y = np.eye(self.num_classes, dtype=np.float32)[cl.astype(int)]
        return x, y