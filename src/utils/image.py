from pydicom import dcmread
from pydicom.pixels import apply_voi_lut
import numpy as np
import cv2


def load_dicom_image(path):
    ds = dcmread(path)
    img2d = ds.pixel_array
    img2d = apply_voi_lut(img2d, ds)

    if ds.PhotometricInterpretation == "MONOCHROME1":
        img2d = np.amax(img2d) - img2d

    img2d = img2d.astype(np.float32)
    return img2d


def crop_to_roi(original):
    img = (original * 255).astype('uint8')
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    _, mask = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    cnts, _ = cv2.findContours(
        mask.astype(
            np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cnt = max(cnts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    return original[y: y + h, x: x + w]
