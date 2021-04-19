from .utils import compute_image_mscn_transform, extract_subband_feats
from .imresize import imresize
import numpy as np
import cv2




def brisque(image):
    y_mscn = compute_image_mscn_transform(image)
    half_scale = imresize(image, scalar_scale = 0.5, method = 'bicubic')
    y_half_mscn = compute_image_mscn_transform(half_scale)
    feats_full = extract_subband_feats(y_mscn)
    feats_half = extract_subband_feats(y_half_mscn)
    return np.concatenate((feats_full, feats_half))


