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


class MEGnetAugSequence(Sequence):
    """
    On-the-fly augmentation for the dual-input MEGnet model.

    Per batch:
      - adds band-limited (<=125 Hz) white noise to temporal_input
      - with p_flip probability PER SAMPLE, multiplies the time series by -1
        and simultaneously inverts the matched spatial map (255 - sp)
    """
    def __init__(self, arrSP, arrTS, arrCL, batch_size,
                 noise_snr=0.1, p_flip=0.5, cutoff_hz=125.0,
                 sfreq=300.0, num_classes=4, shuffle=True, seed=0):
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
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(self.arrTS.shape[0] / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i * self.batch_size:(i + 1) * self.batch_size]

        # copy so we never mutate the source arrays
        sp = self.arrSP[idx].astype(np.float64).copy()
        ts = self.arrTS[idx].astype(np.float64).copy()
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
        sp[flip] = 255.0 - sp[flip]
        sp = np.clip(sp, 0, 255)

        x = {'spatial_input': sp.astype(np.float32),
             'temporal_input': ts.astype(np.float32)}
        y = np.eye(self.num_classes, dtype=np.float32)[cl.astype(int)]
        return x, y