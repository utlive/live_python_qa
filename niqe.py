
from .utils import compute_image_mscn_transform, extract_on_patches

import numpy as np
import scipy.io
import cv2
import os


def niqe(img):
    blocksizerow = 96
    blocksizecol = 96
    h, w = img.shape

    module_path = os.path.dirname(__file__)
    params = scipy.io.loadmat(os.path.join(module_path, 'niqe_nss_parameters.mat'))
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

    scale_percent = 50  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)

    img2 = cv2.resize(img, (height, width), interpolation=cv2.INTER_CUBIC)

    mscn1 = compute_image_mscn_transform(img, extend_mode='nearest')
    mscn1 = mscn1.astype(np.float32)

    mscn2 = compute_image_mscn_transform(img2, extend_mode='nearest')
    mscn2 = mscn2.astype(np.float32)

    feats_lvl1 = extract_on_patches(mscn1, blocksizerow, blocksizecol)
    feats_lvl2 = extract_on_patches(mscn2, blocksizerow/2, blocksizecol/2)

    # stack the scale features
    feats = np.hstack((feats_lvl1, feats_lvl2))

    mu_distparam = np.mean(feats, axis=0)
    cov_distparam = np.cov(feats.T)

    invcov_param = np.linalg.pinv((cov_prisparam + cov_distparam)/2)

    xd = mu_prisparam - mu_distparam
    quality = np.sqrt(np.dot(np.dot(xd, invcov_param), xd.T))[0][0]

    return np.hstack((mu_distparam, [quality]))
