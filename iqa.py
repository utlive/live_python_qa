from .utils import compute_image_mscn_transform,extract_subband_feats
import cv2
import numpy as np

def brisque(image):
    y_mscn = compute_image_mscn_transform(image)
    half_scale = cv2.resize(image, dsize=(0,0),fx=0.5,fy=0.5, interpolation=cv2.INTER_LANCZOS4)
    feats_full = extract_subband_feats(y_mscn)
    feats_half = extract_subband_feats(half_scale)
    return np.concatenate((feats_full,feats_half))
