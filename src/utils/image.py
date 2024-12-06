from pydicom import dcmread
from pydicom.pixels import apply_voi_lut
import numpy as np
import cv2
from src.utils.normalization import truncate_normalization

def crop_to_roi(img: np.array):
    """Crop mammogram to breast region

    Args:
        img (np.array): The original image

    Returns:
        tuple (np.array, np.array): (cropped_image, roi)
    """
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    _, breast_mask = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    cnts, _ = cv2.findContours(
        breast_mask.astype(
            np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cnt = max(cnts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    return img[y: y + h, x: x + w], breast_mask[y: y + h, x: x + w]


def load_dicom_image(path: str):
    """Load a .dicom image file

    Args:
        path (str): Path to the dicom file

    Returns:
        np.array: Loaded image as np.array
    """
    ds = dcmread(path)
    img2d = ds.pixel_array
    img2d = apply_voi_lut(img2d, ds)

    if ds.PhotometricInterpretation == "MONOCHROME1":
        img2d = np.amax(img2d) - img2d
    return cv2.normalize(img2d, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

def get_final_image(path: str, img_size: int):
    original_image = load_dicom_image(path)
    cropped_image, cropped_roi = crop_to_roi(original_image)
    normalized_image = truncate_normalization(cropped_image, cropped_roi)
    resized_image = cv2.resize(
        normalized_image,
        (img_size, img_size),
        interpolation=cv2.INTER_LINEAR,
    )
    return cv2.normalize(resized_image, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
