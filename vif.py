from .utils import moments, im2col

import numpy as np


def vif_spatial(img_ref, img_dist, k=11, sigma_nsq=0.1, padding=None):
    x = img_ref.astype('float32')
    y = img_dist.astype('float32')

    _, _, var_x, var_y, cov_xy = moments(x, y, k, 1)

    g = cov_xy / (var_x + 1e-10)
    sv_sq = var_y - g * cov_xy

    g[var_x < 1e-10] = 0
    sv_sq[var_x < 1e-10] = var_y[var_x < 1e-10]
    var_x[var_x < 1e-10] = 0

    g[var_y < 1e-10] = 0
    sv_sq[var_y < 1e-10] = 0

    sv_sq[g < 0] = var_x[g < 0]
    g[g < 0] = 0
    sv_sq[sv_sq < 1e-10] = 1e-10

    vif_val = np.sum(np.log(1 + g**2 * var_x / (sv_sq + sigma_nsq)) + 1e-4)/np.sum(np.log(1 + var_x / sigma_nsq) + 1e-4)
    return vif_val


def vif_gsm_model(pyr, subband_keys, M):
    tol = 1e-15
    s_all = []
    lamda_all = []

    for subband_key in subband_keys:
        y = pyr[subband_key]
        y_size = (int(y.shape[0]/M)*M, int(y.shape[1]/M)*M)
        y = y[:y_size[0], :y_size[1]]

        y_vecs = im2col(y, M, 1)
        cov = np.cov(y_vecs)
        lamda, V = np.linalg.eigh(cov)
        lamda[lamda < tol] = tol
        cov = V@np.diag(lamda)@V.T

        y_vecs = im2col(y, M, M)

        s = np.linalg.inv(cov)@y_vecs
        s = np.sum(s * y_vecs, 0)/(M*M)
        s = s.reshape((int(y_size[0]/M), int(y_size[1]/M)))

        s_all.append(s)
        lamda_all.append(lamda)

    return s_all, lamda_all


def vif_channel_est(pyr_ref, pyr_dist, subband_keys, M):
    tol = 1e-15
    g_all = []
    sigma_vsq_all = []

    for i, subband_key in enumerate(subband_keys):
        y_ref = pyr_ref[subband_key]
        y_dist = pyr_dist[subband_key]

        lev = int(np.ceil((i+1)/2))
        winsize = 2**lev + 1

        y_size = (int(y_ref.shape[0]/M)*M, int(y_ref.shape[1]/M)*M)
        y_ref = y_ref[:y_size[0], :y_size[1]]
        y_dist = y_dist[:y_size[0], :y_size[1]]

        _, _, var_x, var_y, cov_xy = moments(y_ref, y_dist, winsize, M)

        g = cov_xy / (var_x + tol)
        sigma_vsq = var_y - g*cov_xy

        g[var_x < tol] = 0
        sigma_vsq[var_x < tol] = var_y[var_x < tol]
        var_x[var_x < tol] = 0

        g[var_y < tol] = 0
        sigma_vsq[var_y < tol] = 0

        sigma_vsq[g < 0] = var_y[g < 0]
        g[g < 0] = 0

        sigma_vsq[sigma_vsq < tol] = tol

        g_all.append(g)
        sigma_vsq_all.append(sigma_vsq)

    return g_all, sigma_vsq_all


def vif(img_ref, img_dist, wavelet='steerable'):
    M = 3
    sigma_nsq = 0.1

    if wavelet == 'steerable':
        from pyrtools.pyramids import SteerablePyramidSpace as SPyr
        pyr_ref = SPyr(img_ref, 4, 5, 'reflect1').pyr_coeffs
        pyr_dist = SPyr(img_dist, 4, 5, 'reflect1').pyr_coeffs
        subband_keys = []
        for key in list(pyr_ref.keys())[1:-2:3]:
            subband_keys.append(key)
    else:
        import pywt
        from pywt import wavedec2
        assert wavelet in pywt.wavelist(kind='discrete'), 'Invalid choice of wavelet'
        ret_ref = wavedec2(img_ref, wavelet, 'reflect', 4)
        ret_dist = wavedec2(img_dist, wavelet, 'reflect', 4)
        pyr_ref = {}
        pyr_dist = {}
        subband_keys = []
        for i in range(4):
            pyr_ref[(3-i, 0)] = ret_ref[i+1][0]
            pyr_ref[(3-i, 1)] = ret_ref[i+1][1]
            pyr_dist[(3-i, 0)] = ret_dist[i+1][0]
            pyr_dist[(3-i, 1)] = ret_dist[i+1][1]
            subband_keys.append((3-i, 0))
            subband_keys.append((3-i, 1))
        pyr_ref[4] = ret_ref[0]
        pyr_dist[4] = ret_dist[0]

    subband_keys.reverse()
    n_subbands = len(subband_keys)

    [g_all, sigma_vsq_all] = vif_channel_est(pyr_ref, pyr_dist, subband_keys, M)

    [s_all, lamda_all] = vif_gsm_model(pyr_ref, subband_keys, M)

    nums = np.zeros((n_subbands,))
    dens = np.zeros((n_subbands,))
    for i in range(n_subbands):
        g = g_all[i]
        sigma_vsq = sigma_vsq_all[i]
        s = s_all[i]
        lamda = lamda_all[i]

        n_eigs = len(lamda)

        lev = int(np.ceil((i+1)/2))
        winsize = 2**lev + 1
        offset = (winsize - 1)/2
        offset = int(np.ceil(offset/M))

        g = g[offset:-offset, offset:-offset]
        sigma_vsq = sigma_vsq[offset:-offset, offset:-offset]
        s = s[offset:-offset, offset:-offset]

        for j in range(n_eigs):
            nums[i] += np.mean(np.log(1 + g*g*s*lamda[j]/(sigma_vsq+sigma_nsq)))
            dens[i] += np.mean(np.log(1 + s*lamda[j]/sigma_nsq))

    return np.mean(nums)/np.mean(dens)
