import numpy as np
import math
from scipy.special import gamma
import scipy
import scipy.ndimage


def paired_product(new_im):
    shift1 = np.roll(new_im.copy(), 1, axis=1)
    shift2 = np.roll(new_im.copy(), 1, axis=0)
    shift3 = np.roll(np.roll(new_im.copy(), 1, axis=0), 1, axis=1)
    shift4 = np.roll(np.roll(new_im.copy(), 1, axis=0), -1, axis=1)

    H_img = shift1 * new_im
    V_img = shift2 * new_im
    D1_img = shift3 * new_im
    D2_img = shift4 * new_im

    return (H_img, V_img, D1_img, D2_img)


def gen_gauss_window(lw, sigma):
    sd = np.float32(sigma)
    lw = int(lw)
    weights = [0.0] * (2 * lw + 1)
    weights[lw] = 1.0
    sum = 1.0
    sd *= sd
    for ii in range(1, lw + 1):
        tmp = np.exp(-0.5 * np.float32(ii * ii) / sd)
        weights[lw + ii] = tmp
        weights[lw - ii] = tmp
        sum += 2.0 * tmp
    for ii in range(2 * lw + 1):
        weights[ii] /= sum
    return weights


def estimateggdparam(vec):
    gam = np.asarray([x / 1000.0 for x in range(200, 10000, 1)])
    r_gam = (gamma(1.0/gam)*gamma(3.0/gam))/((gamma(2.0/gam))**2)
    # print(np.mean(vec))
    sigma_sq = np.mean(vec**2) #-(np.mean(vec))**2
    sigma = np.sqrt(sigma_sq)
    E = np.mean(np.abs(vec))
    rho = sigma_sq / (E**2 + 1e-6)
    array_position = (np.abs(rho - r_gam)).argmin()
    alphaparam = gam[array_position]
    return alphaparam, sigma


def compute_image_mscn_transform(image, C=1, avg_window=None, extend_mode='constant'):
    if avg_window is None:
        avg_window = gen_gauss_window(3, 7.0/6.0)
    assert len(np.shape(image)) == 2
    h, w = np.shape(image)
    mu_image = np.zeros((h, w), dtype=np.float32)
    var_image = np.zeros((h, w), dtype=np.float32)
    image = np.array(image).astype('float32')
    scipy.ndimage.correlate1d(image, avg_window, 0, mu_image, mode=extend_mode)
    scipy.ndimage.correlate1d(mu_image, avg_window, 1, mu_image, mode=extend_mode)
    scipy.ndimage.correlate1d(image**2, avg_window, 0, var_image, mode=extend_mode)
    scipy.ndimage.correlate1d(var_image, avg_window, 1, var_image, mode=extend_mode)
    var_image = np.sqrt(np.abs(var_image - mu_image**2))
    return (image - mu_image)/(var_image + C)


def extract_subband_feats(mscncoefs):
    # alpha_m,  = extract_ggd_features(mscncoefs)
    alpha_m, sigma = estimateggdparam(mscncoefs)
    pps1, pps2, pps3, pps4 = paired_product(mscncoefs)
    alpha1, N1, bl1, br1, lsq1, rsq1 = aggd_features(pps1)
    alpha2, N2, bl2, br2, lsq2, rsq2 = aggd_features(pps2)
    alpha3, N3, bl3, br3, lsq3, rsq3 = aggd_features(pps3)
    alpha4, N4, bl4, br4, lsq4, rsq4 = aggd_features(pps4)
    return np.array([
            alpha_m, sigma,
            alpha1, N1, lsq1**2, rsq1**2,  # (V)
            alpha2, N2, lsq2**2, rsq2**2,  # (H)
            alpha3, N3, lsq3**2, rsq3**2,  # (D1)
            alpha4, N4, lsq4**2, rsq4**2,  # (D2)
    ])


def aggd_features(imdata):
    # Flatten imdata
    imdata.shape = (len(imdata.flat),)
    imdata2 = imdata*imdata
    left_data = imdata2[imdata < 0]
    right_data = imdata2[imdata >= 0]
    left_mean_sqrt = 0
    right_mean_sqrt = 0
    if len(left_data) > 0:
        left_mean_sqrt = np.sqrt(np.average(left_data))
    if len(right_data) > 0:
        right_mean_sqrt = np.sqrt(np.average(right_data))

    if right_mean_sqrt != 0:
        gamma_hat = left_mean_sqrt/right_mean_sqrt
    else:
        gamma_hat = np.inf

    # Solve r-hat norm
    imdata2_mean = np.mean(imdata2)
    if imdata2_mean != 0:
        r_hat = (np.average(np.abs(imdata))**2) / (np.average(imdata2))
    else:
        r_hat = np.inf
    rhat_norm = r_hat * (((math.pow(gamma_hat, 3) + 1)*(gamma_hat + 1)) / math.pow(math.pow(gamma_hat, 2) + 1, 2))

    # win = np.array(gen_gauss_window(3, 7.0/6.0))

    gamma_range = np.arange(0.2, 10, 0.001)
    a = scipy.special.gamma(2.0/gamma_range)
    a *= a
    b = scipy.special.gamma(1.0/gamma_range)
    c = scipy.special.gamma(3.0/gamma_range)
    prec_gammas = a/(b*c)

    # solve alpha by guessing values that minimize ro
    pos = np.argmin((prec_gammas - rhat_norm)**2)
    alpha = gamma_range[pos]

    gam1 = scipy.special.gamma(1.0/alpha)
    gam2 = scipy.special.gamma(2.0/alpha)
    gam3 = scipy.special.gamma(3.0/alpha)

    aggdratio = np.sqrt(gam1) / np.sqrt(gam3)
    bl = aggdratio * left_mean_sqrt
    br = aggdratio * right_mean_sqrt

    # Mean parameter
    N = (br - bl)*(gam2 / gam1) #*aggdratio
    return (alpha, N, bl, br, left_mean_sqrt, right_mean_sqrt)


def integral_image(x):
    M, N = x.shape
    int_x = np.zeros((M+1, N+1))
    int_x[1:, 1:] = np.cumsum(np.cumsum(x, 0), 1)
    return int_x


def moments(x, y, k, stride, padding=None):
    kh = kw = k

    k_norm = k**2

    if padding is None:
        x_pad = x
        y_pad = y
    else:
        x_pad = np.pad(x, int((kh - stride)/2), mode=padding)
        y_pad = np.pad(y, int((kh - stride)/2), mode=padding)

    int_1_x = integral_image(x_pad)
    int_1_y = integral_image(y_pad)

    mu_x = (int_1_x[:-kh:stride, :-kw:stride] - int_1_x[:-kh:stride, kw::stride] - int_1_x[kh::stride, :-kw:stride] + int_1_x[kh::stride, kw::stride]) / k_norm
    mu_y = (int_1_y[:-kh:stride, :-kw:stride] - int_1_y[:-kh:stride, kw::stride] - int_1_y[kh::stride, :-kw:stride] + int_1_y[kh::stride, kw::stride]) / k_norm

    int_2_x = integral_image(x_pad**2)
    int_2_y = integral_image(y_pad**2)

    int_xy = integral_image(x_pad*y_pad)

    var_x = (int_2_x[:-kh:stride, :-kw:stride] - int_2_x[:-kh:stride, kw::stride] - int_2_x[kh::stride, :-kw:stride] + int_2_x[kh::stride, kw::stride]) / k_norm - mu_x**2
    var_y = (int_2_y[:-kh:stride, :-kw:stride] - int_2_y[:-kh:stride, kw::stride] - int_2_y[kh::stride, :-kw:stride] + int_2_y[kh::stride, kw::stride]) / k_norm - mu_y**2

    cov_xy = (int_xy[:-kh:stride, :-kw:stride] - int_xy[:-kh:stride, kw::stride] - int_xy[kh::stride, :-kw:stride] + int_xy[kh::stride, kw::stride]) / k_norm - mu_x*mu_y

    # Correcting negative values of variance.
    mask_x = (var_x < 0)
    mask_y = (var_y < 0)

    var_x[mask_x] = 0
    var_y[mask_y] = 0

    # If either variance was negative, it has been set to zero.
    # So, the correponding covariance should also be zero.
    cov_xy[mask_x | mask_y] = 0

    return (mu_x, mu_y, var_x, var_y, cov_xy)


def im2col(img, k, stride=1):
    # Parameters
    m, n = img.shape
    s0, s1 = img.strides
    nrows = m - k + 1
    ncols = n - k + 1
    shape = (k, k, nrows, ncols)
    arr_stride = (s0, s1, s0, s1)

    ret = np.lib.stride_tricks.as_strided(img, shape=shape, strides=arr_stride)
    return ret[:, :, ::stride, ::stride].reshape(k*k, -1)
  

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

def extract_on_patches(img, blocksizerow, blocksizecol):
    h, w = img.shape
    blocksizerow = np.int(blocksizerow)
    blocksizecol = np.int(blocksizecol)
    patches = []
    for j in range(0, np.int(h-blocksizerow+1), np.int(blocksizerow)):
        for i in range(0, np.int(w-blocksizecol+1), np.int(blocksizecol)):
            patch = img[j:j+blocksizerow, i:i+blocksizecol]
            patches.append(patch)

    patches = np.array(patches)
    
    patch_features = []
    for p in patches:
        p_brisque_features = extract_subband_feats(p)
        patch_features.append(p_brisque_features)
    patch_features = np.array(patch_features)

    return patch_features
