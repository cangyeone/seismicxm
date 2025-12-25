import h5py 
import numpy as np 
import torch 
# -*- coding: utf-8 -*-
"""
Seismic event features (36-D + 201-D) per Wang et al. (Computers & Geosciences, 2023).
- Randomly generates 3C waveform (xN, xE, xZ), picks tP, tS, epicentral distance Δ
- Extracts:
  * 36-D engineered features (8 groups)
  * 201-D amplitude spectrum (0~50 Hz, 0.25 Hz step)
Only depends on numpy + standard library. Copy-run friendly.

Author: you
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict

# -----------------------
# Global config & helpers
# -----------------------
SEED = 2025
rng = np.random.default_rng(SEED)

@dataclass
class Waveform:
    xN: np.ndarray
    xE: np.ndarray
    xZ: np.ndarray
    fs: float       # sampling rate (Hz)
    tP: float       # P arrival (s)
    tS: float       # S arrival (s)
    delta_km: float # epicentral distance Δ (km)

def detrend_demean(x: np.ndarray) -> np.ndarray:
    n = x.size
    t = np.arange(n, dtype=np.float64)
    # least squares linear detrend
    A = np.vstack([t, np.ones_like(t)]).T
    a, b = np.linalg.lstsq(A, x, rcond=None)[0]
    y = x - (a * t + b)
    # demean (should be close to zero already)
    y = y - y.mean()
    return y

def cosine_taper(x: np.ndarray, p: float = 0.05) -> np.ndarray:
    """Simple symmetric cosine taper with proportion p on each end."""
    n = x.size
    m = int(np.floor(p * n))
    w = np.ones(n)
    if m > 0:
        k = np.arange(m)
        w[:m] = 0.5 * (1 - np.cos(np.pi * (k + 1) / (m + 1)))
        w[-m:] = w[:m][::-1]
    return x * w

def unit_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    s = np.sqrt((x**2).mean()) + eps
    return x / s

def envelope(x: np.ndarray) -> np.ndarray:
    """Analytic signal via frequency-domain Hilbert transform (no scipy)."""
    n = x.size
    X = np.fft.rfft(x)
    h = np.zeros_like(X, dtype=np.complex128)
    # Construct the equivalent of hilbert multiplier for rfft domain:
    # For real FFT of length n: bins [0..n//2]
    # Double positive freqs except DC and Nyquist (if even)
    h[0] = 1.0
    if n % 2 == 0:
        h[-1] = 1.0
        h[1:-1] = 2.0
    else:
        h[1:] = 2.0
    x_analytic = np.fft.irfft(X * h, n)
    return np.abs(x_analytic)

def zcr(x: np.ndarray) -> float:
    """Zero-crossing rate: sign changes per sample."""
    s = np.sign(x)
    s[s == 0] = 1  # treat zeros as positive to avoid overcount
    changes = np.sum(s[1:] * s[:-1] < 0)
    return changes / max(1, x.size)

def bandpass_rms(spec_f: np.ndarray, spec_amp: np.ndarray, f1: float, f2: float) -> float:
    """RMS amplitude in [f1,f2] from (f, |X(f)|)."""
    mask = (spec_f >= f1) & (spec_f < f2)
    if not np.any(mask):
        return 0.0
    a = spec_amp[mask]
    return np.sqrt((a**2).mean())

def simple_spectrum(x: np.ndarray, fs: float, nfft: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """One-sided amplitude spectrum |X(f)| via FFT."""
    if nfft is None:
        # power of 2 >= len(x)
        n = int(2**np.ceil(np.log2(x.size)))
    else:
        n = nfft
    X = np.fft.rfft(x, n=n)
    f = np.fft.rfftfreq(n, d=1.0/fs)
    amp = np.abs(X) * 2.0 / x.size  # amplitude scaling
    return f, amp

def resample_spectrum_to_grid(f: np.ndarray, amp: np.ndarray, fmin=0.0, fmax=50.0, df=0.25) -> Tuple[np.ndarray, np.ndarray]:
    fgrid = np.arange(fmin, fmax + 1e-9, df)
    agrid = np.interp(fgrid, f, amp, left=0.0, right=0.0)
    return fgrid, agrid

def brune_model_amp(f: np.ndarray, f_c: float, A0: float) -> np.ndarray:
    # |Ω(f)| ~ A0 / sqrt(1 + (f/fc)^2)  -> amplitude form for magnitude spectrum
    return A0 / np.sqrt(1.0 + (f / (f_c + 1e-12))**2)

def jimenez_model_amp(f: np.ndarray, f_c: float, A0: float) -> np.ndarray:
    # |Ω(f)| ~ A0 / (1 + (f/fc)^4)^(1/2)
    return A0 / np.sqrt(1.0 + (f / (f_c + 1e-12))**4)

def fit_corner_frequency(f: np.ndarray, amp: np.ndarray, fmin=0.8, fmax=40.0, n_fc=120, model="brune") -> float:
    """Grid-search fc that minimizes L2 error between |X(f)| and model amp (with optimal A0)."""
    mask = (f >= fmin) & (f <= fmax)
    f_use = f[mask]
    a_use = amp[mask]
    if f_use.size < 8 or np.all(a_use == 0):
        return float('nan')

    fcs = np.geomspace(0.5, 50.0, num=n_fc)
    best_fc = fcs[0]
    best_err = np.inf
    for fc in fcs:
        if model == "brune":
            M = brune_model_amp(f_use, fc, 1.0)
        else:
            M = jimenez_model_amp(f_use, fc, 1.0)
        # optimal A0 via least squares: minimize ||a - A0*M||^2 -> A0 = (M·a)/(M·M)
        denom = (M * M).sum() + 1e-12
        A0 = (M * a_use).sum() / denom
        pred = A0 * M
        err = ((a_use - pred) ** 2).mean()
        if err < best_err:
            best_err = err
            best_fc = fc
    return float(best_fc)

def real_cepstrum_complexity(x: np.ndarray, fs: float, qmin=0.005, qmax=1.0) -> float:
    """
    Cepstral 'complexity':
    - Compute real cepstrum c[q] = IFFT(log|X(f)|)
    - Count peaks above mean+1σ within quefrency window [qmin, qmax] s
    """
    nfft = int(2**np.ceil(np.log2(x.size)))
    f, amp = simple_spectrum(x, fs, nfft=nfft)
    amp = np.maximum(amp, 1e-12)
    log_mag = np.log(amp)
    c = np.fft.irfft(log_mag, n=nfft)
    q = np.arange(c.size) / fs  # quefrency axis (s)
    mask = (q >= qmin) & (q <= qmax)
    if not np.any(mask):
        return 0.0
    segment = c[mask]
    mu, sigma = segment.mean(), segment.std(ddof=1) + 1e-12
    thr = mu + 1.0 * sigma
    # simple local maxima count above threshold
    cnt = 0
    for i in range(1, segment.size - 1):
        if segment[i] > segment[i-1] and segment[i] > segment[i+1] and segment[i] > thr:
            cnt += 1
    return float(cnt)

def instantaneous_frequency_complexity(x: np.ndarray, fs: float) -> float:
    """
    IF 'complexity' from analytic signal:
    - Compute instantaneous phase φ(t) from analytic signal
    - IF f_inst = (fs/2π) * unwrap(diff(φ))
    - Return normalized variability: std(f_inst) / (mean(|f_inst|)+eps)
    """
    # analytic signal (same helper used by envelope)
    n = x.size
    X = np.fft.rfft(x)
    h = np.zeros_like(X, dtype=np.complex128)
    h[0] = 1.0
    if n % 2 == 0:
        h[-1] = 1.0
        h[1:-1] = 2.0
    else:
        h[1:] = 2.0
    xa = np.fft.irfft(X * h, n)
    # phase
    phi = np.unwrap(np.angle(xa + 1e-18))
    dphi = np.diff(phi)
    f_inst = (fs / (2.0 * np.pi)) * dphi
    if f_inst.size < 8:
        return 0.0
    num = np.std(f_inst, ddof=1)
    den = np.mean(np.abs(f_inst)) + 1e-12
    return float(num / den)

def find_tcoda(x: np.ndarray, fs: float, pre_noise_dur=5.0, tail_min_dur=3.0, thresh_sigma=1.2) -> float:
    """
    Define tcoda as the first time (from end to start) where envelope remains below (mean+thresh_sigma*std) of pre-event noise
    for a continuous duration >= tail_min_dur.
    Returns tcoda in seconds relative to trace start.
    """
    env = envelope(x)
    n_pre = int(pre_noise_dur * fs)
    noise = env[:max(n_pre,1)]
    thr = noise.mean() + thresh_sigma * (noise.std(ddof=1) + 1e-12)

    below = env < thr
    run = 0
    need = int(tail_min_dur * fs)
    idx = x.size - 1
    while idx >= 0:
        if below[idx]:
            run += 1
            if run >= need:
                # continuous below-threshold segment found; tcoda at segment start
                start_idx = idx + run - 1
                # walk forward to first point below threshold to be conservative
                # (already there), convert to seconds
                return float(start_idx / fs)
        else:
            run = 0
        idx -= 1
    # fallback: end of trace
    return float((x.size - 1) / fs)

def window_around(arr: np.ndarray, fs: float, t0: float, pre: float, post: float) -> np.ndarray:
    i0 = int(np.round(t0 * fs))
    i1 = max(0, i0 - int(pre * fs))
    i2 = min(arr.size, i0 + int(post * fs))
    seg = arr[i1:i2].copy()
    if seg.size < 4:
        return seg
    # preprocess per paper: de-average, detrend, taper, normalize
    seg = detrend_demean(seg)
    seg = cosine_taper(seg, p=0.05)
    seg = unit_normalize(seg)
    return seg

def p_s_components(xN: np.ndarray, xE: np.ndarray, xZ: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple component projection:
    - P dominated by vertical (Z)
    - S dominated by horizontal (sqrt(N^2 + E^2))
    """
    P = xZ
    S = np.sqrt(xN**2 + xE**2)
    return P, S

def max_abs(x: np.ndarray) -> float:
    return float(np.max(np.abs(x))) if x.size > 0 else 0.0

# -----------------------
# Feature extraction core
# -----------------------
def extract_features(wf: Waveform) -> Dict[str, np.ndarray]:
    fs = wf.fs
    # windows (seconds) around picks — can调参
    preP, postP = 0.5, 3.0
    preS, postS = 0.5, 6.0

    # Build P- and S- windows from 3C
    Psig = window_around(wf.xZ, fs, wf.tP, preP, postP)   # Pg ~ Z
    Sh = np.sqrt(wf.xN**2 + wf.xE**2)
    Ssig = window_around(Sh, fs, wf.tS, preS, postS)      # Sg ~ horizontal

    # 0) Basic spectra for later ratios
    fP, aP = simple_spectrum(Psig, fs)
    fS, aS = simple_spectrum(Ssig, fs)

    # 1) P/S amplitude ratios
    # 1.1 time-domain max ratio
    ps_max_ratio = (max_abs(Psig) / (max_abs(Ssig) + 1e-12))

    # 1.2 spectral RMS ratio per 1-Hz bands from 1 to 20 Hz
    band_ratios = []
    for fband in range(1, 21):
        rp = bandpass_rms(fP, aP, fband, fband + 1.0)
        rs = bandpass_rms(fS, aS, fband, fband + 1.0)
        band_ratios.append(rp / (rs + 1e-12))
    band_ratios = np.asarray(band_ratios, dtype=np.float64)

    # 2) High/low frequency energy ratio for P and S
    def band_energy_ratio(f, a):
        hf = bandpass_rms(f, a, 5.0, 18.0)
        lf = bandpass_rms(f, a, 0.05, 5.0)
        return hf / (lf + 1e-12)
    hf_lf_P = band_energy_ratio(fP, aP)
    hf_lf_S = band_energy_ratio(fS, aS)

    # 3) Corner frequencies (Brune & Jimenez) for P and S
    fc_brune_P = fit_corner_frequency(fP, aP, model="brune")
    fc_brune_S = fit_corner_frequency(fS, aS, model="brune")
    fc_jim_P   = fit_corner_frequency(fP, aP, model="jimenez")
    fc_jim_S   = fit_corner_frequency(fS, aS, model="jimenez")

    # 4) Waveform duration: t = (tcoda - tP) / Δ
    #    (Use whole Z trace for coda detection)
    tcoda = find_tcoda(wf.xZ, fs)
    duration_norm = (tcoda - wf.tP) / (wf.delta_km + 1e-9)

    # 5) Waveform complexity C for P and S per Eq.(4):
    #    C = ( ∫_{t1}^{t2} y^2 dt / (t2-t1) ) / ( ∫_{tP}^{t_coda} y^2 dt / (t_coda - tP) )
    #    Use each phase window as [t1,t2]; tail energy from phase pick to its coda proxy.
    def complexity(seg: np.ndarray, t_pick: float, use_Z: bool) -> float:
        # phase-window energy density
        if seg.size < 4:
            return 0.0
        e1 = (seg**2).mean()

        # build tail segment from original Z (or H) starting at pick until min(tcoda, pick+20s)
        t_end = min(tcoda, t_pick + 20.0)
        arr = wf.xZ if use_Z else np.sqrt(wf.xN**2 + wf.xE**2)
        i1 = int(np.round(t_pick * fs))
        i2 = int(np.round(t_end * fs))
        i2 = max(i2, i1 + 1)
        tail = arr[i1:i2].copy()
        tail = detrend_demean(tail)
        tail = cosine_taper(tail, p=0.05)
        tail = unit_normalize(tail)
        e2 = (tail**2).mean()
        return float(e1 / (e2 + 1e-12))

    C_P = complexity(Psig, wf.tP, use_Z=True)
    C_S = complexity(Ssig, wf.tS, use_Z=False)

    # 6) Zero-crossing rate for P and S
    ZCR_P = zcr(Psig)
    ZCR_S = zcr(Ssig)

    # 7) Cepstrum complexity for P and S
    Cep_P = real_cepstrum_complexity(Psig, fs)
    Cep_S = real_cepstrum_complexity(Ssig, fs)

    # 8) Instantaneous frequency (IF) complexity for P and S
    IF_P = instantaneous_frequency_complexity(Psig, fs)
    IF_S = instantaneous_frequency_complexity(Ssig, fs)

    # Compose 36-D feature vector (order: 21 ratios + 2(H/L) + 4(fc) + 1(duration) + 2(C) + 2(ZCR) + 2(Cep) + 2(IF))
    feat36 = np.concatenate([
        np.array([ps_max_ratio]),          # 1
        band_ratios,                       # 20  => total 21
        np.array([hf_lf_P, hf_lf_S]),      # +2  => 23
        np.array([fc_brune_P, fc_brune_S, fc_jim_P, fc_jim_S]),  # +4 => 27
        np.array([duration_norm]),         # +1  => 28
        np.array([C_P, C_S]),              # +2  => 30
        np.array([ZCR_P, ZCR_S]),          # +2  => 32
        np.array([Cep_P, Cep_S]),          # +2  => 34
        np.array([IF_P, IF_S])             # +2  => 36
    ]).astype(np.float64)

    # 201-D amplitude spectrum (network-averaged in paper; here single-station demo on Z as example)
    # You may average across multiple stations externally if needed.
    fZ, aZ = simple_spectrum(wf.xZ, fs)
    fgrid, agrid = resample_spectrum_to_grid(fZ, aZ, fmin=0.0, fmax=50.0, df=0.25)
    spec201 = agrid  # length should be 201 (0..50 inclusive step 0.25)

    return {
        "feature_36d": feat36,
        "spectrum_fgrid": fgrid,
        "spectrum_201d": spec201,
        "debug": {
            "tcoda": tcoda,
            "fc_brune_P": fc_brune_P, "fc_brune_S": fc_brune_S,
            "fc_jim_P": fc_jim_P, "fc_jim_S": fc_jim_S
        }
    }
def prep_trace(x):
    x = detrend_demean(x)
    x = cosine_taper(x, p=0.02)
    x = unit_normalize(x)
    return x
# -----------------------
# Synthetic data generator
# -----------------------
def synth_waveform(fs=100.0, dur=120.0) -> Waveform:
    n = int(dur * fs)
    t = np.arange(n) / fs

    # Random picks and distance per paper ranges
    tP = rng.uniform(10.0, 30.0)
    tS = tP + rng.uniform(3.0, 12.0)
    delta_km = rng.uniform(20.0, 400.0)

    # Source-like pulses (Ricker wavelets) centered at tP and tS
    def ricker(t, f0, t0, A=1.0):
        pi2f2 = (np.pi * f0)**2
        tau2 = (t - t0)**2
        return A * (1.0 - 2.0 * pi2f2 * tau2) * np.exp(-pi2f2 * tau2)

    # Create 3C with P on Z (higher freq), S on H (lower freq), plus coda and noise
    xZ = np.zeros_like(t)
    xN = np.zeros_like(t)
    xE = np.zeros_like(t)

    # P: more high-frequency on Z
    xZ += ricker(t, f0=rng.uniform(6, 10), t0=tP, A=rng.uniform(0.8, 1.2))

    # S: stronger on H, lower freq, longer coda
    sH = ricker(t, f0=rng.uniform(2.5, 6), t0=tS, A=rng.uniform(1.0, 1.6))
    # Distribute S energy to N/E with random azimuth
    theta = rng.uniform(0, 2*np.pi)
    xN += sH * np.cos(theta)
    xE += sH * np.sin(theta)

    # Simple coda: exponentially decaying noise after max(P,S)
    t_on = int(min(n-1, int((tS + 1.0) * fs)))
    decay = np.exp(-np.maximum(0, np.arange(n) - t_on) / (fs * rng.uniform(0.8, 2.5)))
    codaZ = decay * rng.normal(0, 0.15, size=n)
    codaH = decay * rng.normal(0, 0.18, size=n)
    xZ += codaZ
    xN += codaH * np.cos(theta)
    xE += codaH * np.sin(theta)

    # Pre-event noise
    noiseZ = rng.normal(0, 0.03, size=n)
    noiseN = rng.normal(0, 0.03, size=n)
    noiseE = rng.normal(0, 0.03, size=n)
    xZ += noiseZ
    xN += noiseN
    xE += noiseE

    # Basic preprocessing on whole traces (detrend, taper, normalize per component)
    def prep_trace(x):
        x = detrend_demean(x)
        x = cosine_taper(x, p=0.02)
        x = unit_normalize(x)
        return x

    xZ = prep_trace(xZ)
    xN = prep_trace(xN)
    xE = prep_trace(xE)

    return Waveform(xN=xN, xE=xE, xZ=xZ, fs=fs, tP=tP, tS=tS, delta_km=delta_km)


if __name__ == "__main__":
    picker = torch.jit.load("large/rnn.origdiff.pnsn.jit")
    file_name = f"data/type.h5"
    with open("large/event.type", "r") as f:
        etype_dict = eval(f.read())
    usage = "train"
    h5keys = np.load(f"data/trainkeys.npz")[usage]
    h5file = h5py.File(file_name, "r")
    infos = []
    for typekey, ekey in h5keys:
        event = h5file[typekey][ekey] 
        typeid = etype_dict[typekey] 
        elon = event.attrs["lon"]
        elat = event.attrs["lat"]
        edep = event.attrs["dep"]
        emag = event.attrs["mag"]
        #for tkey in event.attrs:
        #    print(tkey, event.attrs[tkey])
        station_datas = []
        for skey in event:
            station = event[skey]
            slon, slat, sele = station.attrs["lon"], station.attrs["lat"], station.attrs["dep"]
            dist = station.attrs["dist"]
            if dist>80:continue 
            data = station[:]
            phase_time = {}
            with torch.no_grad():
                phase = picker(torch.from_numpy(data.T).float())
                for c, t, p in phase.cpu().numpy():
                    c = int(c) 
                    if c==0:
                        phase_time["P"] = t
                    elif c==1:
                        phase_time["S"] = t
            if "P" not in phase_time or "S" not in phase_time:continue 
            data = data.astype(np.float64)
            wf = Waveform(xN=prep_trace(data[0, :]), xE=prep_trace(data[1, :]), xZ=prep_trace(data[2, :]), fs=100, tP=phase_time["P"]/100, tS=phase_time["S"]/100, delta_km=dist)
            feats = extract_features(wf)

            f36 = feats["feature_36d"]
            infos.append([feats, typeid, ekey, skey])
    torch.save(infos, f"odata/feature.{usage}.pth")
