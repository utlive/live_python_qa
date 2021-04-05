from .utils import compute_image_mscn_transform, extract_subband_feats
from .utils import extract_on_patches
from .utils import moments
from .utils import vif_channel_est, vif_gsm_model
import cv2
import numpy as np


# TODO: Compare with Matlab
def brisque(image):
    y_mscn = compute_image_mscn_transform(image)
    half_scale = cv2.resize(image, dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LANCZOS4)
    y_half_mscn = compute_image_mscn_transform(half_scale)
    feats_full = extract_subband_feats(y_mscn)
    feats_half = extract_subband_feats(y_half_mscn)
    return np.concatenate((feats_full, feats_half))


def ssim(img_ref, img_dist, k=11, max_val=1, K1=0.01, K2=0.03, no_lum=False, full=False, padding=None, stride=1):
    x = img_ref.astype('float32')
    y = img_dist.astype('float32')
    mu_x, mu_y, var_x, var_y, cov_xy = moments(x, y, k, stride, padding=padding)

    C1 = (max_val*K1)**2
    C2 = (max_val*K2)**2

    if not no_lum:
        l = (2*mu_x*mu_y + C1)/(mu_x**2 + mu_y**2 + C1)
    cs = (2*cov_xy + C2)/(var_x + var_y + C2)

    ssim_map = cs
    if not no_lum:
        ssim_map *= l

    if (full):
        return (np.mean(ssim_map), ssim_map)
    else:
        return np.mean(ssim_map)


def ms_ssim(img_ref, img_dist, k=11, max_val=1, K1=0.01, K2=0.03, full=False, padding=None, stride=1):
    x = img_ref.astype('float32')
    y = img_dist.astype('float32')

    n_levels = 5
    weights = np.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    scores = np.ones((n_levels,))
    for i in range(n_levels-1):
        if np.min(x.shape) <= k:
            break
        scores[i] = ssim(x, y, k, max_val, K1, K2, no_lum=True, padding=padding, stride=stride)
        x = x[:(x.shape[0]//2)*2, :(x.shape[1]//2)*2]
        y = y[:(y.shape[0]//2)*2, :(y.shape[1]//2)*2]
        x = (x[::2, ::2] + x[1::2, ::2] + x[1::2, 1::2] + x[::2, 1::2])/4
        y = (y[::2, ::2] + y[1::2, ::2] + y[1::2, 1::2] + y[::2, 1::2])/4

    if np.min(x.shape) > k:
        scores[-1] = ssim(x, y, k, max_val, K1, K2, no_lum=False, padding=padding, stride=stride)
    msssim = np.prod(np.power(scores, weights))
    if full:
        return msssim, scores
    else:
        return msssim


def vif_spatial(img_ref, img_dist, k=11, max_val=1, sigma_nsq=0.1, padding=None):
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

def niqe(img):
    blocksizerow = 96
    blocksizecol = 96
    h, w = img.shape

    module_path = dirname(__file__)
    params = scipy.io.loadmat(join(module_path, 'niqe_nss_parameters.mat'))
    mu_prisparam = params['mu_prisparam']
    cov_prisparam = params['cov_prisparam']
    if (h < blocksizerow) or (w < blocksizecol):
        print("Input frame is too small")
        exit(0)

    # ensure that the patch divides evenly into img
    hoffset = (h % blocksizerow)
    woffset = (w % blocksizecol)

    if hoffset > 0: 
        img = img[:-hoffset, :]
    if woffset > 0:
        img = img[:, :-woffset]

    img = img.astype(np.float32)

    scale_percent = 50 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100) 

    img2 = cv2.resize(img, (height,width), interpolation=cv2.INTER_CUBIC)

    mscn1, var, mu = compute_image_mscn_transform(img, extend_mode='nearest')
    mscn1 = mscn1.astype(np.float32)

    mscn2, _, _ = compute_image_mscn_transform(img2, extend_mode='nearest')
    mscn2 = mscn2.astype(np.float32)

    feats_lvl1 = extract_on_patches(mscn1, blocksizerow, blocksizecol)
    feats_lvl2 = extract_on_patches(mscn2, blocksizerow/2, blocksizecol/2)

    # stack the scale features
    feats = np.hstack((feats_lvl1, feats_lvl2))# feats_lvl3))

    mu_distparam = np.mean(feats, axis=0)
    cov_distparam = np.cov(feats.T)

    invcov_param = np.linalg.pinv((cov_prisparam + cov_distparam)/2)

    xd = mu_prisparam - mu_distparam 
    quality = np.sqrt(np.dot(np.dot(xd, invcov_param), xd.T))[0][0]

    return np.hstack((mu_distparam, [quality]))
