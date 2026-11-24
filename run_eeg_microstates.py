#!/usr/bin/python3
"""
"""
import contextlib
import io
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")                                                  
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.signal import butter, filtfilt
from scipy.stats import chi2, chi2_contingency, mannwhitneyu, wilcoxon

# --------------------------------------------------------------------------- #
# CALM-Brain settings /home/calm-brain/vsuresh for all data and outputs
# --------------------------------------------------------------------------- #
BASE = Path("/home/calm-brain/vsuresh/eeg_feature_extraction")
SCZ_CSV = BASE / "cohort-level_metrics" / "scz_subject_selection.csv"     # SCZ and HC
DEM_CSV = BASE / "cohort-level_metrics" / "d_source_sessions.csv"         # DEM 
OUT_PATH = BASE / "all_microstates"

PREFIX = {"SCZ": "scz", "DEM": "dem", "HC": "hc"}
CONDITIONS = ("eyes-open", "eyes-closed")   
SHORT = {"eyes-open": "eo", "eyes-closed": "ec"}

# retained from original
BAND = (1.0, 35.0)                   
MODE = "kmeans"                      
N_MAPS = 4
INTERPOL = True                      
L_MAX = 100                          
N_MC = 10                            
ALPHA = 0.01                        
KMAX = 6                             
BLOCK_SIZE = 5000                    
SEED = 0                             
EPOCH_LEN, TRIM, REJECT_UV = 2.0, 2.0, 150.0                       # as in source_alpha.py
MAX_SECONDS = 120                    
MIN_CLEAN_SECONDS = 30
EXCLUDE_CHANNELS = ()               
WORKERS = 2
CCOL = {"eyes-open": "#fdae61", "eyes-closed": "#459cb4"}           
GCOL = {"DEM": "#1C4A99", "SCZ": "#dc097d", "HC": "#12d155"}
MEASURES = {"gev_total": "GEV (total)", "gfp_peaks_per_s": "GFP peaks per s", "mean_lifetime_ms": "mean duration (ms)",
            "entropy": "entropy (nats)", "entropy_rate": "entropy rate (nats)", "relaxation_time": "relaxation time",
            "aif_peak1_ms": "AIF first peak (ms)", "aif_lags_above_markov": "AIF lags above Markov band"}

mne.set_log_level("WARNING")


# =========================================================================== #
# from iriginal
# =========================================================================== #
def bp_filter(data, f_lo, f_hi, fs):
    data_filt = np.zeros_like(data)
    f_ny = fs/2.  
    b_lo = f_lo/f_ny  
    b_hi = f_hi/f_ny  
    p_lp = {"N":6, "Wn":b_hi, "btype":"lowpass", "analog":False, "output":"ba"}
    p_hp = {"N":6, "Wn":b_lo, "btype":"highpass", "analog":False, "output":"ba"}
    bp_b1, bp_a1 = butter(**p_lp)
    bp_b2, bp_a2 = butter(**p_hp)
    data_filt = filtfilt(bp_b1, bp_a1, data, axis=0)
    data_filt = filtfilt(bp_b2, bp_a2, data_filt, axis=0)
    return data_filt


def topo(data, xy, n_grid=64):
    X, Y = xy[:, 0], xy[:, 1]
    r = max([max(np.abs(X)), max(np.abs(Y))])
    Xi = np.linspace(-r,r,n_grid)
    Yi = np.linspace(-r,r,n_grid)
    data_ip = griddata((X, Y), data, (Xi[None,:], Yi[:,None]), method='cubic')
    return data_ip


def eeg2map(data, xy):
    n_grid = 64
    top = topo(data, xy, n_grid)
    mn = np.nanmin(top)
    mx = np.nanmax(top)
    top_norm = (top-mn)/(mx-mn)
    return top_norm


def clustering(data, fs, chs, locs, mode, n_clusters, n_win=3, \
               interpol=True, doplot=False):
    n_ch = data.shape[1]

    data_norm = data - data.mean(axis=1, keepdims=True)
    data_norm /= data_norm.std(axis=1, keepdims=True)

    gfp = np.nanstd(data, axis=1)
    gfp2 = np.sum(gfp**2) 
    gfp_peaks = locmax(gfp)
    data_cluster = data[gfp_peaks,:]
    data_cluster_norm = data_cluster - data_cluster.mean(axis=1, keepdims=True)
    data_cluster_norm /= data_cluster_norm.std(axis=1, keepdims=True)
    print("[+] Data format for clustering [GFP peaks, channels]: {:d} x {:d}"\
         .format(data_cluster.shape[0], data_cluster.shape[1]))

    start = time.time()

    if (mode == "aahc"):
        print("\n[+] Clustering algorithm: AAHC.")
        maps = aahc(data, n_clusters, doplot=False)

    elif (mode == "kmeans"):
        print("\n[+] Clustering algorithm: mod. K-MEANS.")
        maps = kmeans(data, n_maps=n_clusters, n_runs=5)

    elif (mode == "pca"):
        print("\n[+] Clustering algorithm: PCA.")
        # adapted: sklearn.decomposition.PCA(...).fit(data_cluster_norm).components_, computed with numpy
        Xc = data_cluster_norm - data_cluster_norm.mean(axis=0, keepdims=True)
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        maps = np.array([Vt[k,:] for k in range(n_clusters)])

    else:
        raise ValueError(f"clustering mode {mode!r} not available here (use aahc, kmeans or pca)")  

    end = time.time()
    delta_t = end - start
    print(f"[+] Computation time: {delta_t:.2f} sec")

    print("\n[+] Microstate back-fitting:")
    print("data_norm: ", data_norm.shape)
    print("data_cluster_norm: ", data_cluster_norm.shape)
    print("maps: ", maps.shape)

    if interpol:
        C = np.dot(data_cluster_norm, maps.T)/n_ch
        L_gfp = np.argmax(C**2, axis=1) 
        del C
        n_t = data_norm.shape[0]
        # adapted: same nearest-GFP-peak assignment as original loop, vectorised (original is O(n_t x n_peaks))
        nxt = np.clip(np.searchsorted(gfp_peaks, np.arange(n_t)), 0, len(gfp_peaks) - 1)
        prv = np.clip(nxt - 1, 0, len(gfp_peaks) - 1)
        use_prev = np.abs(np.arange(n_t) - gfp_peaks[prv]) <= np.abs(np.arange(n_t) - gfp_peaks[nxt])
        L = L_gfp[np.where(use_prev, prv, nxt)].astype('int')
    else:
        C = np.dot(data_norm, maps.T)/n_ch
        L = np.argmax(C**2, axis=1)
        del C

    maps_norm = maps - maps.mean(axis=1, keepdims=True)
    maps_norm /= maps_norm.std(axis=1, keepdims=True)

    C = np.dot(data_norm, maps_norm.T)/n_ch

    gev = np.zeros(n_clusters)
    for k in range(n_clusters):
        r = L==k
        gev[k] = np.sum(gfp[r]**2 * C[r,k]**2)/gfp2
    print(f"\n[+] Global explained variance GEV = {gev.sum():.3f}")
    for k in range(n_clusters):
        print(f"GEV_{k:d}: {gev[k]:.3f}")

    return maps, L, gfp_peaks, gev


def backfit(data, maps, interpol=True):
# fitting EO and EC data to same maps
    n_ch = data.shape[1]
    n_clusters = maps.shape[0]

    data_norm = data - data.mean(axis=1, keepdims=True)
    data_norm /= data_norm.std(axis=1, keepdims=True)

    gfp = np.nanstd(data, axis=1)
    gfp2 = np.sum(gfp**2) 
    gfp_peaks = locmax(gfp)
    data_cluster = data[gfp_peaks,:]
    data_cluster_norm = data_cluster - data_cluster.mean(axis=1, keepdims=True)
    data_cluster_norm /= data_cluster_norm.std(axis=1, keepdims=True)

    if interpol:
        C = np.dot(data_cluster_norm, maps.T)/n_ch
        L_gfp = np.argmax(C**2, axis=1)
        del C
        n_t = data_norm.shape[0]
        nxt = np.clip(np.searchsorted(gfp_peaks, np.arange(n_t)), 0, len(gfp_peaks) - 1)
        prv = np.clip(nxt - 1, 0, len(gfp_peaks) - 1)
        use_prev = np.abs(np.arange(n_t) - gfp_peaks[prv]) <= np.abs(np.arange(n_t) - gfp_peaks[nxt])
        L = L_gfp[np.where(use_prev, prv, nxt)].astype('int')
    else:
        C = np.dot(data_norm, maps.T)/n_ch
        L = np.argmax(C**2, axis=1)
        del C

    maps_norm = maps - maps.mean(axis=1, keepdims=True)
    maps_norm /= maps_norm.std(axis=1, keepdims=True)
    C = np.dot(data_norm, maps_norm.T)/n_ch
    gev = np.zeros(n_clusters)
    for k in range(n_clusters):
        r = L==k
        gev[k] = np.sum(gfp[r]**2 * C[r,k]**2)/gfp2
    print(f"[+] Global explained variance GEV = {gev.sum():.3f}")
    return L, gfp_peaks, gev


def aahc(data, N_clusters, doplot=False):
    """Atomize and Agglomerative Hierarchical Clustering Algorithm
    AAHC (Murray et al., Brain Topography, 2008)

    Args:
        data: EEG data to cluster, numpy.array (n_samples, n_channels)
        N_clusters: desired number of clusters
        doplot: boolean, plot maps
    Returns:
        maps: n_maps x n_channels (numpy.array)
    """

    def extract_row(A, k):
        v = A[k,:]
        A_ = np.vstack((A[:k,:],A[k+1:,:]))
        return A_, v

    def extract_item(A, k):
        a = A[k]
        A_ = A[:k] + A[k+1:]
        return A_, a

    nt, nch = data.shape

    # --- get GFP peaks ---
    gfp = data.std(axis=1)
    gfp_peaks = locmax(gfp)
    gfp2 = np.sum(gfp**2) # normalizing constant in GEV

    # --- initialize clusters ---
    maps = data[gfp_peaks,:]
    # --- store original gfp peaks and indices ---
    cluster_data = data[gfp_peaks,:]
    n_maps = maps.shape[0]
    print(f"[+] Initial number of clusters: {n_maps:d}\n")

    # --- cluster indices w.r.t. original size, normalized GFP peak data ---
    Ci = [[k] for k in range(n_maps)]

    # --- main loop: atomize + agglomerate ---
    while (n_maps > N_clusters):
        # --- correlations of the data sequence with each cluster ---
        m_x, s_x = data.mean(axis=1, keepdims=True), data.std(axis=1)
        m_y, s_y = maps.mean(axis=1, keepdims=True), maps.std(axis=1)
        s_xy = 1.*nch*np.outer(s_x, s_y)
        C = np.dot(data-m_x, np.transpose(maps-m_y)) / s_xy

        # --- microstate sequence, ignore polarity ---
        L = np.argmax(C**2, axis=1)

        # --- GEV (global explained variance) of cluster k ---
        gev = np.zeros(n_maps)
        for k in range(n_maps):
            r = L==k
            gev[k] = np.sum(gfp[r]**2 * C[r,k]**2)/gfp2

        # --- merge cluster with the minimum GEV ---
        imin = np.argmin(gev)

        # --- N => N-1 ---
        maps, _ = extract_row(maps, imin)
        Ci, reC = extract_item(Ci, imin)
        re_cluster = []  # indices of updated clusters
        for k in reC:  # map index to re-assign
            c = cluster_data[k,:]
            m_x, s_x = maps.mean(axis=1, keepdims=True), maps.std(axis=1)
            m_y, s_y = c.mean(), c.std()
            s_xy = 1.*nch*s_x*s_y
            C = np.dot(maps-m_x, c-m_y)/s_xy
            inew = np.argmax(C**2) # ignore polarity
            re_cluster.append(inew)
            Ci[inew].append(k)
        n_maps = len(Ci)

        # --- update clusters ---
        re_cluster = list(set(re_cluster)) # unique list of updated clusters

        # re-clustering by eigenvector method
        for i in re_cluster:
            idx = Ci[i]
            Vt = cluster_data[idx,:]
            Sk = np.dot(Vt.T, Vt)
            evals, evecs = np.linalg.eig(Sk)
            c = evecs[:, np.argmax(np.abs(evals))]
            c = np.real(c)
            maps[i] = c/np.sqrt(np.sum(c**2))

    print()
    return maps


def kmeans(data, n_maps, n_runs=10, maxerr=1e-6, maxiter=500):
    n_t = data.shape[0]
    n_ch = data.shape[1]
    data = data - data.mean(axis=1, keepdims=True)

    # GFP peaks
    gfp = np.std(data, axis=1)
    gfp_peaks = locmax(gfp)
    gfp_values = gfp[gfp_peaks]
    gfp2 = np.sum(gfp_values**2) # normalizing constant in GEV
    n_gfp = gfp_peaks.shape[0]

    # clustering of GFP peak maps only
    V = data[gfp_peaks, :]
    sumV2 = np.sum(V**2)

    # store results for each k-means run
    cv_list =   []  # cross-validation criterion for each k-means run
    gev_list =  []  # GEV of each map for each k-means run
    gevT_list = []  # total GEV values for each k-means run
    maps_list = []  # microstate maps for each k-means run
    L_list =    []  # microstate label sequence for each k-means run
    for run in range(n_runs):
        # initialize random cluster centroids (indices w.r.t. n_gfp)
        rndi = np.random.permutation(n_gfp)[:n_maps]
        maps = V[rndi, :]
        # normalize row-wise (across EEG channels)
        maps /= np.sqrt(np.sum(maps**2, axis=1, keepdims=True))
        # initialize
        n_iter = 0
        var0 = 1.0
        var1 = 0.0
        # convergence criterion: variance estimate (step 6)
        while ( (np.abs((var0-var1)/var0) > maxerr) & (n_iter < maxiter) ):
            # (step 3) microstate sequence (= current cluster assignment)
            C = np.dot(V, maps.T)
            C /= (n_ch*np.outer(gfp[gfp_peaks], np.std(maps, axis=1)))
            L = np.argmax(C**2, axis=1)
            # (step 4)
            for k in range(n_maps):
                Vt = V[L==k, :]
                # (step 4a)
                Sk = np.dot(Vt.T, Vt)
                # (step 4b)
                evals, evecs = np.linalg.eig(Sk)
                v = evecs[:, np.argmax(np.abs(evals))]
                v = v.real
                maps[k, :] = v/np.sqrt(np.sum(v**2))
            # (step 5)
            var1 = var0
            var0 = sumV2 - np.sum(np.sum(maps[L, :]*V, axis=1)**2)
            var0 /= (n_gfp*(n_ch-1))
            n_iter += 1
        if (n_iter < maxiter):
            print((f"\tK-means run {run+1:d}/{n_runs:d} converged after "
                   f"{n_iter:d} iterations."))
        else:
            print((f"\tK-means run {run+1:d}/{n_runs:d} did NOT converge "
                   f"after {maxiter:d} iterations."))

        # CROSS-VALIDATION criterion for this run (step 8)
        C_ = np.dot(data, maps.T)
        C_ /= (n_ch*np.outer(gfp, np.std(maps, axis=1)))
        L_ = np.argmax(C_**2, axis=1)
        var = np.sum(data**2) - np.sum(np.sum(maps[L_, :]*data, axis=1)**2)
        var /= (n_t*(n_ch-1))
        cv = var * (n_ch-1)**2/(n_ch-n_maps-1.)**2

        # GEV (global explained variance) of cluster k
        gev = np.zeros(n_maps)
        for k in range(n_maps):
            r = L==k
            gev[k] = np.sum(gfp_values[r]**2 * C[r,k]**2)/gfp2
        gev_total = np.sum(gev)

        # store
        cv_list.append(cv)
        gev_list.append(gev)
        gevT_list.append(gev_total)
        maps_list.append(maps)
        L_list.append(L_)

    # select best run
    k_opt = np.argmin(cv_list)
    maps = maps_list[k_opt]

    return maps


def p_empirical(data, n_clusters):
    p = np.zeros(n_clusters)
    n = len(data)
    for i in range(n):
        p[data[i]] += 1.0
    p /= n
    return p


def T_empirical(data, n_clusters):
    T = np.zeros((n_clusters, n_clusters))
    n = len(data)
    for i in range(n-1):
        T[data[i], data[i+1]] += 1.0
    p_row = np.sum(T, axis=1)
    for i in range(n_clusters):
        if ( p_row[i] != 0.0 ):
            for j in range(n_clusters):
                T[i,j] /= p_row[i]  
    return T


def max_entropy(n_clusters):
    h_max = np.log(float(n_clusters))
    return h_max


def surrogate_mc(p, T, n_clusters, n):
    psum = np.cumsum(p)
    Tsum = np.cumsum(T, axis=1)
    # initial state according to p:
    mc = [np.min(np.argwhere(np.random.rand() < psum))]
    # next state according to T:
    for i in range(1, n):
        mc.append(np.min(np.argwhere(np.random.rand() < Tsum[mc[i-1]])))
    return np.array(mc)


def mutinf(x, ns, lmax):
    """
    *** Symbols must be 0, 1, 2, ... to use as indices directly! ***"""
    n = len(x)
    mi = np.zeros(lmax)
    for l in range(lmax):
        nmax = n-l
        h1 = H_1(x[:nmax], ns)
        h2 = H_1(x[l:l+nmax], ns)
        h12 = H_2(x[:nmax], x[l:l+nmax], ns)
        mi[l] = h1 + h2 - h12
    return mi                                                        


def mutinf_CI(p, T, n, alpha, nrep, lmax):
    ns = len(p)
    mi_array = np.zeros((nrep,lmax))
    for r in range(nrep):
        print(f"surrogate MC # {r+1:d}/{nrep:d}")
        x_mc = surrogate_mc(p, T, ns, n)
        mi_mc = mutinf(x_mc, ns, lmax)
        mi_array[r,:] = mi_mc
    return mi_array


def excess_entropy_rate(x, ns, kmax, doplot=False):
    # y = ax+b: line fit to joint entropy for range of histories k
    # a = entropy rate (slope)
    # b = excess entropy (intersect.)
    h_ = np.zeros(kmax)
    for k in range(kmax): h_[k] = H_k(x, ns, k+1)
    ks = np.arange(1,kmax+1)
    a, b = np.polyfit(ks, h_, 1)
    return (a, b)


def mc_entropy_rate(p, T):
    h = 0.
    for i, j in np.ndindex(T.shape):
        if (T[i,j] > 0):
            h -= ( p[i]*T[i,j]*np.log(T[i,j]) )
    return h


def aif_peak1(mi, fs, doplot=False):
    dt = 1000./fs # sampling interval [ms]
    mi_filt = np.convolve(mi, np.ones(3)/3., mode='same')
    mx0 = 8 # 8
    jmax = mx0 + locmax(mi_filt[mx0:])[0]
    mx_mi = dt*jmax
    return jmax, mx_mi


def H_1(x, ns):
    n = len(x)
    p = np.bincount(x, minlength=ns).astype(float) # symbol distribution
    p /= n
    h = -np.sum(p[p>0]*np.log(p[p>0]))
    return h


def H_2(x, y, ns):
    n = min([len(x), len(y)])
    p = np.bincount(x[:n]*ns + y[:n], minlength=ns*ns).reshape(ns, ns).astype(float) # joint distribution
    p /= n
    h = -np.sum(p[p>0]*np.log(p[p>0]))
    return h


def H_k(x, ns, k):
    """Shannon's joint entropy from x[n+p:n-m]
    x: symbolic time series
    ns: number of symbols
    k: length of k-history
    """
    N = len(x)
    f = np.zeros(tuple(k*[ns]))
    for t in range(N-k): f[tuple(x[t:t+k])] += 1.0
    f /= (N-k) # normalize distribution
    hk = -np.sum(f[f>0]*np.log(f[f>0]))
    return hk


def testMarkov0(x, ns, alpha, verbose=True):
    if verbose:
        print("ZERO-ORDER MARKOVIANITY:")
    n = len(x)
    f_ij = np.zeros((ns,ns))
    f_i = np.zeros(ns)
    f_j = np.zeros(ns)
    # calculate f_ij p( x[t]=i, p( x[t+1]=j ) )
    for t in range(n-1):
        i = x[t]
        j = x[t+1]
        f_ij[i,j] += 1.0
        f_i[i] += 1.0
        f_j[j] += 1.0
    T = 0.0 # statistic
    for i, j in np.ndindex(f_ij.shape):
        f = f_ij[i,j]*f_i[i]*f_j[j]
        if (f > 0):
            num_ = n*f_ij[i,j]
            den_ = f_i[i]*f_j[j]
            T += (f_ij[i,j] * np.log(num_/den_))
    T *= 2.0
    df = (ns-1.0) * (ns-1.0)
    p = chi2.sf(T, df, loc=0, scale=1)
    if verbose:
        print(f"p: {p:.2e} | t: {T:.3f} | df: {df:.1f}")
    return p


def testMarkov1(X, ns, alpha, verbose=True):
    if verbose:
        print("\nFIRST-ORDER MARKOVIANITY:")
    n = len(X)
    f_ijk = np.zeros((ns,ns,ns))
    f_ij = np.zeros((ns,ns))
    f_jk = np.zeros((ns,ns))
    f_j = np.zeros(ns)
    for t in range(n-2):
        i = X[t]
        j = X[t+1]
        k = X[t+2]
        f_ijk[i,j,k] += 1.0
        f_ij[i,j] += 1.0
        f_jk[j,k] += 1.0
        f_j[j] += 1.0
    T = 0.0
    for i, j, k in np.ndindex(f_ijk.shape):
        f = f_ijk[i][j][k]*f_j[j]*f_ij[i][j]*f_jk[j][k]
        if (f > 0):
            num_ = f_ijk[i,j,k]*f_j[j]
            den_ = f_ij[i,j]*f_jk[j,k]
            T += (f_ijk[i,j,k]*np.log(num_/den_))
    T *= 2.0
    df = ns*(ns-1)*(ns-1)
    p = chi2.sf(T, df, loc=0, scale=1)
    if verbose:
        print(f"p: {p:.2e} | t: {T:.3f} | df: {df:.1f}")
    return p


def testMarkov2(X, ns, alpha, verbose=True):
    if verbose:
        print("\nSECOND-ORDER MARKOVIANITY:")
    n = len(X)
    f_ijkl = np.zeros((ns,ns,ns,ns))
    f_ijk = np.zeros((ns,ns,ns))
    f_jkl = np.zeros((ns,ns,ns))
    f_jk = np.zeros((ns,ns))
    for t in range(n-3):
        i = X[t]
        j = X[t+1]
        k = X[t+2]
        l = X[t+3]
        f_ijkl[i,j,k,l] += 1.0
        f_ijk[i,j,k] += 1.0
        f_jkl[j,k,l] += 1.0
        f_jk[j,k] += 1.0
    T = 0.0
    for i, j, k, l in np.ndindex(f_ijkl.shape):
        f = f_ijkl[i,j,k,l]*f_ijk[i,j,k]*f_jkl[j,k,l]*f_jk[j,k]
        if (f > 0):
            num_ = f_ijkl[i,j,k,l]*f_jk[j,k]
            den_ = f_ijk[i,j,k]*f_jkl[j,k,l]
            T += (f_ijkl[i,j,k,l]*np.log(num_/den_))
    T *= 2.0
    df = ns*ns*(ns-1)*(ns-1)
    p = chi2.sf(T, df, loc=0, scale=1)
    if verbose:
        print(f"p: {p:.2e} | t: {T:.3f} | df: {df:.1f}")
    return p


def conditionalHomogeneityTest(X, ns, l, alpha, verbose=True):
    """Test conditional homogeneity of non-overlapping blocks of
    length l of symbolic sequence X with ns symbols
    cf. Kullback, Technometrics (1962), Table 9.1."""
    if verbose:
        print("\nCONDITIONAL HOMOGENEITY (three-way table):")
    n = len(X)
    r = int(np.floor(float(n)/float(l))) # number of blocks
    if verbose:
        print("Split data in r = {:d} blocks of length {:d}.".format(r,l))
    f_ijk = np.zeros((r,ns,ns))
    f_ij = np.zeros((r,ns))
    f_jk = np.zeros((ns,ns))
    f_i = np.zeros(r)
    f_j = np.zeros(ns)

    # calculate f_ijk (time / block dep. transition matrix)
    for i in  range(r): # block index
        for ii in range(l-1): # pos. inside the current block
            j = X[i*l + ii]
            k = X[i*l + ii + 1]
            f_ijk[i,j,k] += 1.0
            f_ij[i,j] += 1.0
            f_jk[j,k] += 1.0
            f_i[i] += 1.0
            f_j[j] += 1.0

    # conditional homogeneity (Markovianity stationarity)
    T = 0.0
    for i, j, k in np.ndindex(f_ijk.shape):
        # conditional homogeneity
        f = f_ijk[i,j,k]*f_j[j]*f_ij[i,j]*f_jk[j,k]
        if (f > 0):
            num_ = f_ijk[i,j,k]*f_j[j]
            den_ = f_ij[i,j]*f_jk[j,k]
            T += (f_ijk[i,j,k]*np.log(num_/den_))
    T *= 2.0
    df = (r-1)*(ns-1)*ns
    p = chi2.sf(T, df, loc=0, scale=1)
    if verbose:
        print(f"p: {p:.2e} | t: {T:.3f} | df: {df:.1f}")
    return p


def symmetryTest(X, ns, alpha, verbose=True):
    """Test symmetry of the transition matrix of symbolic sequence X with
    ns symbols, cf. Kullback, Technometrics (1962)"""
    if verbose:
        print("\nSYMMETRY:")
    n = len(X)
    f_ij = np.zeros((ns,ns))
    for t in range(n-1):
        i = X[t]
        j = X[t+1]
        f_ij[i,j] += 1.0
    T = 0.0
    for i, j in np.ndindex(f_ij.shape):
        if (i != j):
            f = f_ij[i,j]*f_ij[j,i]
            if (f > 0):
                num_ = 2*f_ij[i,j]
                den_ = f_ij[i,j]+f_ij[j,i]
                T += (f_ij[i,j]*np.log(num_/den_))
    T *= 2.0
    df = ns*(ns-1)/2
    p = chi2.sf(T, df, loc=0, scale=1)
    if verbose:
        print(f"p: {p:.2e} | t: {T:.3f} | df: {df:.1f}")
    return p


def geoTest(X, ns, dt, alpha, verbose=True):
    """Test the geometric distribution of lifetime distributions.
    Returns: p_values for each symbol"""
    if verbose:
        print("\nGEOMETRIC DISTRIBUTION of lifetimes:\n")
    tauDist = lifetimes(X, ns)
    T = T_empirical(X, ns)
    p_values = np.zeros(ns)
    for s in range(ns): # test for each symbol
        if verbose:
            print(f"\nTesting the distribution of symbol # {s:d}")
        # m = max_tau:
        m = len(tauDist[s])
        # observed lifetime distribution:
        q_obs = np.zeros(m)
        # theoretical lifetime distribution:
        q_exp = np.zeros(m)
        for j in range(m):
            # observed frequency of lifetime j+1 for state s
            q_obs[j] = tauDist[s][j]
            # expected frequency
            q_exp[j] = (1.0-T[s][s]) * (T[s][s]**j)
        q_exp *= sum(q_obs)

        t = 0.0 # chi2 statistic
        for j in range(m):
            if ((q_obs[j] > 0) & (q_exp[j] > 0)):
                t += (q_obs[j]*np.log(q_obs[j]/q_exp[j]))
        t *= 2.0
        df = m-1
        p0 = chi2.sf(t, df, loc=0, scale=1)
        if verbose:
            print(f"p: {p0:.2e} | t: {t:.3f} | df: {df:.1f}")
        p_values[s] = p0

        if verbose:
            try:   
                g1, p1, dof1, expctd1 = chi2_contingency(np.vstack((q_obs,q_exp)), lambda_='log-likelihood')
                print((f"G-test (log-likelihood) p: {p1:.2e}, g: {g1:.3f}, df: {dof1:.1f}"))
            except ValueError as e:
                print(f"G-test (log-likelihood) not computed: {e}")
    return p_values


def lifetimes(X, ns):
    """Compute the lifetime distributions for each symbol
    in a symbolic sequence X with ns symbols.
    Returns: tauDist: list of lists, lifetime distributions for each symbol"""
    n = len(X)
    tauList = [[] for i in range(ns)] # unsorted lifetimes, [[],[],[],[]]
    i = 0 # current index
    s = X[i] # current symbol
    tau = 1.0 # current lifetime
    while (i < n-1):
        i += 1
        if ( X[i] == s ):
            tau += 1.0
        else:
            tauList[int(s)].append(tau)
            s = X[i]
            tau = 1.0
    tauList[int(s)].append(tau) # last state
    # find max lifetime for each symbol
    tau_max = [max(L) if L else 1 for L in tauList]                   
    tauDist = [] # empty histograms for each symbol
    for i in range(ns): tauDist.append( [0.]*int(tau_max[i]) )
    # lifetime distributions
    for s in range(ns):
        for j in range(len(tauList[s])):
            tau = tauList[s][j]
            tauDist[s][int(tau)-1] += 1.0
    return tauDist


def mixing_time(X, ns):
    """Relaxation time, inverse of spectral gap"""
    T_hat = T_empirical(X,ns) # transition matrix
    ev = np.linalg.eigvals(T_hat)
    ev = np.real(ev)
    ev.sort() # ascending
    ev2 = np.flipud(ev) # descending
    sg = ev2[0] - ev2[1] # spectral gap
    T_mix = 1.0/sg # mixing time
    return T_mix


def multiple_comparisons(p_values, method):
    """Multiple comparisons correction.
    adapted: Bonferroni ('b', their tutorial's choice) without statsmodels."""
    if method != "b":
        raise ValueError("only Bonferroni ('b') is available without statsmodels")
    p = np.asarray(p_values, float)
    return np.minimum(p * np.isfinite(p).sum(), 1.0)


def locmax(x):
    """Get local maxima of 1D-array"""
    dx = np.diff(x) # discrete 1st derivative
    zc = np.diff(np.sign(dx)) # zero-crossings of dx
    m = 1 + np.where(zc == -2)[0] # indices of local max.
    return m


# =========================================================================== #
# Batch runner for the cohorts 
# =========================================================================== #
def session_list():
    cols = ["session_id", "group", "file", "eo_start", "eo_end", "ec_start", "ec_end"]
    scz = pd.read_csv(SCZ_CSV)
    scz = scz[scz.included.astype(bool) & scz.group.isin(["SCZ", "HC"])]
    dem = pd.read_csv(DEM_CSV)
    dem = dem[dem.included.astype(bool)] if "included" in dem else dem[dem.status.eq("ok")]
    dem = dem[dem.group.eq("DEM")]
    dem = (dem.assign(subject=dem.session_id.str.extract(r"(sub-[^_]+)", expand=False))
              .sort_values("session_id").drop_duplicates("subject"))      # one session per subject
    s = pd.concat([scz[cols], dem[cols]], ignore_index=True).drop_duplicates("session_id")
    order = {g: k for k, g in enumerate(PREFIX)}
    return s.sort_values(["group", "session_id"], key=lambda c: c.map(order) if c.name == "group" else c,
                         ignore_index=True)


def prep(raw):
    names = [c for c in raw.ch_names if c.startswith("E") and c[1:].isdigit() and c != "E129"]
    raw.pick(names)
    raw.info["bads"] = [c for c in raw.ch_names if c in set(raw.info["bads"]) | set(EXCLUDE_CHANNELS)]
    if not any("average" in p["desc"] for p in raw.info["projs"]):
        raw.set_eeg_reference("average", projection=True)
    raw.apply_proj()
    return raw


def clean_data(raw, start, end):
    sf = raw.info["sfreq"]
    seg = raw.copy().crop(start + TRIM, min(end - TRIM, raw.times[-1]), include_tmax=False)
    good = [c for c in seg.ch_names if c not in seg.info["bads"]]
    ep = mne.make_fixed_length_epochs(seg, duration=EPOCH_LEN, preload=True)
    ep.drop_bad(reject=dict(eeg=REJECT_UV * 1e-6))
    kept = [b for b, log in enumerate(ep.drop_log) if len(log) == 0]
    if MAX_SECONDS is not None:
        kept = kept[:int(MAX_SECONDS // EPOCH_LEN)]
    n = int(round(EPOCH_LEN * sf))
    filt = bp_filter(seg.get_data(picks=good).T * 1e6, BAND[0], BAND[1], sf)        # their filter, whole window
    data = np.concatenate([filt[b * n:(b + 1) * n] for b in kept]) if kept else np.empty((0, len(good)))
    n_joins = int(sum(1 for a, b in zip(kept[:-1], kept[1:]) if b != a + 1))
    return data, good, n_joins


def electrode_xy(names):
    """Azimuthal equidistant projection of the template GSN-HydroCel-128 positions (x = right, y = front)."""
    pos = mne.channels.make_standard_montage("GSN-HydroCel-128").get_positions()["ch_pos"]
    allp = np.array([pos[f"E{i}"] for i in range(1, 129)])
    A = np.c_[2 * allp, np.ones(len(allp))]                          # least-squares sphere fit
    centre = np.linalg.lstsq(A, (allp ** 2).sum(1), rcond=None)[0][:3]
    v = np.array([pos[c] for c in names]) - centre
    theta = np.arccos(np.clip(v[:, 2] / np.linalg.norm(v, axis=1), -1, 1))
    phi = np.arctan2(v[:, 1], v[:, 0])
    return np.c_[theta * np.cos(phi), theta * np.sin(phi)]


def switch_distribution(x):
    a, b = x[:-1], x[1:]
    m = a != b
    c = np.bincount(a[m] * N_MAPS + b[m], minlength=N_MAPS * N_MAPS).astype(float)
    return c / c.sum() if c.sum() else c


def sequence_stats(x, gfp_peaks, gev, fs):
    row = {}
    dt = 1000. / fs
    secs = len(x) / fs

    p_hat = p_empirical(x, N_MAPS)
    T_hat = T_empirical(x, N_MAPS)
    taus = lifetimes(x, N_MAPS)
    dur = np.full(N_MAPS, np.nan)
    for i in range(N_MAPS):
        n_seg = float(np.sum(taus[i]))
        if p_hat[i] > 0 and n_seg > 0:
            dur[i] = float(np.sum(dt * np.arange(1, len(taus[i]) + 1) * np.array(taus[i]) / n_seg))
        row[f"coverage_{i}"], row[f"duration_ms_{i}"] = p_hat[i], dur[i]
        row[f"occurrence_per_s_{i}"] = n_seg / secs if p_hat[i] > 0 else 0.0
        row[f"gev_{i}"] = gev[i]
    n_segments = 1 + int(np.count_nonzero(x[1:] != x[:-1]))
    row.update(gev_total=gev.sum(), gfp_peaks_per_s=len(gfp_peaks) / secs, mean_lifetime_ms=len(x) / n_segments * dt,
               relaxation_time=mixing_time(x, N_MAPS))

    h_rate, _ = excess_entropy_rate(x, N_MAPS, KMAX)
    row.update(entropy=H_1(x, N_MAPS), entropy_max=max_entropy(N_MAPS), entropy_rate=h_rate,
               mc_entropy_rate=mc_entropy_rate(p_hat, T_hat))

    print("\n[7] Markovianity tests:\n")
    row.update(p_markov0=testMarkov0(x, N_MAPS, ALPHA), p_markov1=testMarkov1(x, N_MAPS, ALPHA),
               p_markov2=testMarkov2(x, N_MAPS, ALPHA))
    for i, p in enumerate(geoTest(x, N_MAPS, dt, ALPHA)):
        row[f"p_geometric_{i}"] = p
    row["p_stationarity"] = (conditionalHomogeneityTest(x, N_MAPS, BLOCK_SIZE, ALPHA)
                             if len(x) >= 2 * BLOCK_SIZE else np.nan)
    row["p_symmetry"] = symmetryTest(x, N_MAPS, ALPHA)

    print(f"\n[14] Mutual information, {N_MC} Markov surrogates")
    aif = mutinf(x, N_MAPS, L_MAX)
    aif_array = mutinf_CI(p_hat, T_hat, len(x), ALPHA, N_MC, L_MAX)
    pct = np.percentile(aif_array, [100. * ALPHA / 2., 100. * (1 - ALPHA / 2.)], axis=0)
    try:
        _, row["aif_peak1_ms"] = aif_peak1(aif, fs)
    except IndexError:                                               # adapted: no local maximum after 8 lags
        row["aif_peak1_ms"] = np.nan
    row["aif_lags_above_markov"] = int((aif > pct[1]).sum())
    return row, dict(x=x, gev=gev, p=p_hat, T=T_hat, dur=dur, aif=aif, pct=pct, switches=switch_distribution(x))


def subject_figure(prefix, sid, group, maps, names, res, fs, change):
    xy = electrode_xy(names)
    eo, ec = res["eyes-open"], res["eyes-closed"]
    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(3, N_MAPS, height_ratios=[1, .9, 1], hspace=.45)
    for k in range(N_MAPS):
        ax = fig.add_subplot(gs[0, k])
        ax.imshow(eeg2map(maps[k], xy), cmap=plt.cm.seismic, origin="lower")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"map {k}\nGEV EO {eo['gev'][k]:.2f}, EC {ec['gev'][k]:.2f}", fontsize=9)
    x, w = np.arange(N_MAPS), .38
    half = N_MAPS // 2
    for col, (key, label) in enumerate((("p", "coverage (share of time)"), ("dur", "mean duration (ms)"))):
        ax = fig.add_subplot(gs[1, col * half:(col + 1) * half])
        ax.bar(x - w / 2, eo[key], w, color=CCOL["eyes-open"], label="eyes open")
        ax.bar(x + w / 2, ec[key], w, color=CCOL["eyes-closed"], label="eyes closed")
        ax.set_xticks(x, [f"map {k}" for k in x])
        ax.set_title(label, fontsize=10)
        ax.legend(fontsize=8)
    ax = fig.add_subplot(gs[2, :])
    lag = np.arange(L_MAX) * 1000. / fs
    for cond in CONDITIONS:
        d = res[cond]
        ax.semilogy(lag, d["aif"], "-", color=CCOL[cond], lw=2, label=f"AIF, {cond}")
        ax.fill_between(lag, d["pct"][0], d["pct"][1], color=CCOL[cond], alpha=.25, lw=0, label=f"Markov band, {cond}")
    ax.set_xlabel("time lag [ms]")
    ax.set_ylabel("AIF [nats]")
    ax.legend(fontsize=8, ncol=2)
    fig.suptitle(f"{group} {sid}: one set of maps for eyes open and eyes closed  "
                 f"(coverage change {change[0]:.2f}, switch change {change[1]:.2f})", fontsize=12)
    fig.savefig(OUT_PATH / f"{prefix}_{sid}_eo_vs_ec.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def run_session(r):
    sid, prefix, group = r["session_id"], PREFIX[r["group"]], r["group"]
    np.random.seed(SEED)
    rows = {c: dict(session_id=sid, group=group, condition=c, status="ok") for c in CONDITIONS}
    log = io.StringIO()
    try:
        with contextlib.redirect_stdout(log):
            raw = prep(mne.io.read_raw_fif(r["file"], preload=True))
            fs = raw.info["sfreq"]
            data, n_joins = {}, {}
            for c in CONDITIONS:
                start, end = (r["eo_start"], r["eo_end"]) if c == "eyes-open" else (r["ec_start"], r["ec_end"])
                if pd.isna(start):
                    raise ValueError(f"no saved {c} window")
                data[c], names, n_joins[c] = clean_data(raw, start, end)
            n = min(len(d) for d in data.values())
            if n / fs < MIN_CLEAN_SECONDS:
                raise ValueError(f"only {n / fs:.0f} s of clean data in one condition")
            data = {c: d[:n] for c, d in data.items()}              # the same amount of data from each condition

            print("===== clustering eyes open + eyes closed together =====")
            maps, _, _, gev_pool = clustering(np.concatenate([data[c] for c in CONDITIONS]), fs, names, None,
                                              MODE, N_MAPS, interpol=INTERPOL)
            order = np.argsort(gev_pool)[::-1]                       # maps sorted by GEV (no manual A-D)
            maps, gev_pool = maps[order], gev_pool[order]

            res = {}
            for c in CONDITIONS:
                print(f"\n===== {c}: back-fitting the shared maps =====")
                x, peaks, gev = backfit(data[c], maps, INTERPOL)
                row, res[c] = sequence_stats(x, peaks, gev, fs)
                rows[c].update(row, seconds=n / fs, n_joins=n_joins[c], n_good_channels=len(names))
                d = res[c]
                np.savez_compressed(OUT_PATH / f"{prefix}_{sid}_{SHORT[c]}.npz", maps=maps, channels=np.array(names),
                                    labels=d["x"].astype(np.int8), gev=d["gev"], coverage=d["p"], T_hat=d["T"],
                                    aif=d["aif"], aif_markov_band=d["pct"], lag_ms=np.arange(L_MAX) * 1000. / fs)

            sw_eo, sw_ec = res["eyes-open"]["switches"], res["eyes-closed"]["switches"]
            change = (0.5 * float(np.abs(res["eyes-closed"]["p"] - res["eyes-open"]["p"]).sum()),
                      0.5 * float(np.abs(sw_ec - sw_eo).sum()) if sw_eo.sum() and sw_ec.sum() else np.nan)  # no switches: undefined
            C = np.corrcoef(maps)
            for c in CONDITIONS:
                rows[c].update(coverage_change=change[0], switch_change=change[1], gev_pooled_total=gev_pool.sum(),
                               cmax_map_correlation=float(np.max(np.abs(C - np.diag(np.diag(C))))))
            subject_figure(prefix, sid, group, maps, names, res, fs, change)
    except Exception as e:                                           # noqa: BLE001
        for c in CONDITIONS:
            rows[c]["status"] = f"error: {type(e).__name__}: {e}"
    (OUT_PATH / f"{prefix}_{sid}_log.txt").write_text(log.getvalue())
    return list(rows.values())


def eo_vs_ec(summary):
    """Cohort comparisons: EO vs EC for the sequence measures (paired), mean AIF curves, and map-use change."""
    ok = summary[summary.status.eq("ok")]
    wide = ok.pivot_table(index=["session_id", "group"], columns="condition", values=list(MEASURES))
    groups = [g for g in PREFIX if g in set(ok.group)]
    stats_rows = []

    fig, axes = plt.subplots(2, 4, figsize=(18, 8.5))
    for ax, (m, label) in zip(axes.ravel(), MEASURES.items()):
        for i, g in enumerate(groups):
            d = wide[m].xs(g, level="group")[["eyes-open", "eyes-closed"]].dropna()
            x0 = 3 * i
            if d.empty:
                continue
            ax.boxplot([d["eyes-open"], d["eyes-closed"]], positions=[x0, x0 + 1], widths=.6, showfliers=False,
                       medianprops=dict(color="k"))
            for a, b in d.to_numpy():
                ax.plot([x0, x0 + 1], [a, b], color="grey", alpha=.3, lw=.7)
            ax.scatter(np.full(len(d), x0), d["eyes-open"], s=10, color=CCOL["eyes-open"], zorder=3)
            ax.scatter(np.full(len(d), x0 + 1), d["eyes-closed"], s=10, color=CCOL["eyes-closed"], zorder=3)
            diff = d["eyes-closed"] - d["eyes-open"]
            p = wilcoxon(d["eyes-open"], d["eyes-closed"]).pvalue if len(d) >= 5 and (diff != 0).any() else np.nan
            stats_rows.append(dict(group=g, measure=m, n=len(d), median_eo=d["eyes-open"].median(),
                                   median_ec=d["eyes-closed"].median(), median_ec_minus_eo=diff.median(), p_wilcoxon=p))
            ax.text(x0 + .5, 1.01, f"p = {p:.2g}", transform=ax.get_xaxis_transform(), ha="center", fontsize=8)
        ax.set_xticks([3 * i + k for i in range(len(groups)) for k in (0, 1)],
                      [f"{g}\nEO" if k == 0 else f"{g}\nEC" for g in groups for k in (0, 1)], fontsize=8)
        ax.set_title(label, fontsize=10, pad=14)
    fig.suptitle("Microstates, eyes open (orange) vs eyes closed (blue); lines join each subject's two values; "
                 "Wilcoxon signed-rank p (uncorrected)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_PATH / "microstate_eo_vs_ec_measures.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    st = pd.DataFrame(stats_rows)
    if not st.empty:
        st["p_bonf"] = multiple_comparisons(st.p_wilcoxon.to_numpy(), "b")
        st.to_csv(OUT_PATH / "microstate_eo_vs_ec_stats.csv", index=False)

    fig, axes = plt.subplots(1, len(groups), figsize=(6 * len(groups), 4.5), squeeze=False, sharey=True)
    for ax, g in zip(axes[0], groups):
        for cond in CONDITIONS:
            curves, lag = [], None
            for s in ok.session_id[ok.group.eq(g) & ok.condition.eq(cond)]:
                d = np.load(OUT_PATH / f"{PREFIX[g]}_{s}_{SHORT[cond]}.npz")
                curves.append(d["aif"])
                lag = d["lag_ms"]
            if not curves:
                continue
            c = np.array(curves)
            mu = c.mean(0)
            se = c.std(0, ddof=1) / np.sqrt(len(c)) if len(c) > 1 else np.zeros(c.shape[1])
            ax.semilogy(lag, mu, color=CCOL[cond], lw=2, label=f"{cond} (n = {len(c)})")
            ax.fill_between(lag, np.maximum(mu - se, 1e-6), mu + se, color=CCOL[cond], alpha=.25, lw=0)
        ax.set_title(g)
        ax.set_xlabel("time lag [ms]")
        ax.legend(fontsize=8)
    axes[0, 0].set_ylabel("AIF [nats]")
    fig.suptitle("Microstate autoinformation, eyes open vs eyes closed (mean +/- SEM)", fontsize=11)
    fig.savefig(OUT_PATH / "microstate_eo_vs_ec_aif.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    sub = ok[ok.condition.eq("eyes-closed")]                         # one row per subject
    rng = np.random.default_rng(0)
    change_rows = []
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, (m, label) in zip(axes, (("coverage_change", "change in time spent in each map, EO -> EC"),
                                     ("switch_change", "change in switching between maps, EO -> EC"))):
        vals = [sub[m][sub.group.eq(g)].dropna() for g in groups]
        ax.boxplot(vals, positions=range(len(groups)), widths=.6, showfliers=False, medianprops=dict(color="k"))
        for i, (g, v) in enumerate(zip(groups, vals)):
            ax.scatter(i + rng.uniform(-.12, .12, len(v)), v, s=14, color=GCOL[g], alpha=.7, zorder=3)
            p = (mannwhitneyu(v, vals[groups.index("HC")]).pvalue
                 if g != "HC" and "HC" in groups and len(v) >= 3 and len(vals[groups.index("HC")]) >= 3 else np.nan)
            change_rows.append(dict(measure=m, group=g, n=len(v), median=v.median(), p_vs_hc_mannwhitney=p))
            if g != "HC":
                ax.text(i, 1.01, f"vs HC p = {p:.2g}", transform=ax.get_xaxis_transform(), ha="center", fontsize=8)
        ax.set_xticks(range(len(groups)), [f"{g}\n(n = {len(v)})" for g, v in zip(groups, vals)])
        ax.set_ylabel("0 = same in EO and EC, 1 = completely different")
        ax.set_title(label, fontsize=10, pad=14)
    fig.suptitle("How much map use changes between eyes open and eyes closed (each subject's own maps); "
                 "Mann-Whitney vs HC, uncorrected", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT_PATH / "microstate_change.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(change_rows).to_csv(OUT_PATH / "microstate_change_stats.csv", index=False)
    return st


def main():
    OUT_PATH.mkdir(parents=True, exist_ok=True)
    sessions = session_list()
    print("subjects:", ", ".join(f"{g} {n}" for g, n in sessions.group.value_counts().reindex(list(PREFIX)).items()))
    print(f"band {BAND} Hz, mode {MODE}, {N_MAPS} maps, interpol={INTERPOL}, up to {MAX_SECONDS} s per condition")
    rows = []
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        for k, res in enumerate(ex.map(run_session, sessions.to_dict("records")), 1):
            for r in res:
                print(f"{k}/{len(sessions)} {r['group']} {r['session_id']} {r['condition']}: {r['status']}", flush=True)
            rows += res
    summary = pd.DataFrame(rows)

    ok = summary.status.eq("ok")
    for col in [c for c in summary.columns if c.startswith("p_") and not c[2:].isdigit()]:
        for cond in CONDITIONS:
            m = ok & summary.condition.eq(cond) & summary[col].notna()
            summary.loc[m, col + "_bonf"] = multiple_comparisons(summary.loc[m, col].to_numpy(), "b")
    summary.to_csv(OUT_PATH / "microstate_summary.csv", index=False)
    if len(CONDITIONS) == 2:
        st = eo_vs_ec(summary)
        if not st.empty:
            print("\neyes open vs eyes closed (Wilcoxon signed-rank)")
            print(st.round(3).to_string(index=False))

    cols = ["gev_total", "gfp_peaks_per_s", "entropy", "entropy_rate", "aif_peak1_ms", "aif_lags_above_markov"]
    if ok.any():
        print("\ncohort means")
        print(summary[ok].groupby(["condition", "group"])[cols].mean().round(3).to_string())
    print(f"done -> {OUT_PATH}")


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    main()
