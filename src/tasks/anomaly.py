import os
import cv2
import logging
from tqdm import tqdm
from src.utils.image import load_dicom_image
from concurrent.futures import ProcessPoolExecutor
from src.utils.dataframe import prepare_vindr_finding_dataframe
from src.utils.image import crop_to_roi
from src.utils.normalization import truncate_normalization


def prepare_row(row, data_dir: str, out_dir: str, img_size: int):
    try:
        sample_path = os.path.join(
            data_dir, 'images', row['study_id'], row['image_id'] + '.dicom')
        original_image = load_dicom_image(sample_path)
        cropped_image, cropped_roi = crop_to_roi(original_image)
        normalized_image = truncate_normalization(cropped_image, cropped_roi)
        resized_image = cv2.resize(
            normalized_image,
            (img_size, img_size),
            interpolation=cv2.INTER_LINEAR,
        )
        new_class = '0_normal' if row['finding_categories'] == 'no_finding' else '1_abnormal'
        output_image_path = os.path.join(
            out_dir, new_class, "{}.png".format(row.name))
        cv2.imwrite(output_image_path, resized_image)
    except Exception as e:
        img_id = row['image_id']
        logging.error(f'Failed to process image {img_id}: {e}')


def prepare_anomaly_dataset(data_dir: str, out_dir: str, img_size: int, class_list: list):
    """Prepare the VINDR MAMMO dataset for anomaly specific classification (binary classification)

    Args:
        data_dir (str): Path to original cbis dataset
        out_dir (str): Path to save the prepared cbis dataset
        img_size (int): New image size
        class_list (list): List of the classes to keep
    """
    train_df = prepare_vindr_finding_dataframe(data_dir, class_list, True)
    test_df = prepare_vindr_finding_dataframe(data_dir, class_list, False)

    os.makedirs(os.path.join(out_dir, 'train', '0_normal'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'test', '1_abnormal'), exist_ok=True)

    train_out_dir = os.path.join(out_dir, 'train')
    test_out_dir = os.path.join(out_dir, 'test')

    with ProcessPoolExecutor() as executor:
        list(
            tqdm(
                executor.map(
                    prepare_row,
                    [row for _, row in train_df.iterrows()],
                    [data_dir] * len(train_df),
                    [train_out_dir] * len(train_df),
                    [img_size] * len(train_df),
                ),
                total=len(train_df),
            )
        )

    with ProcessPoolExecutor() as executor:
        list(
            tqdm(
                executor.map(
                    prepare_row,
                    [row for _, row in test_df.iterrows()],
                    [data_dir] * len(test_df),
                    [test_out_dir] * len(test_df),
                    [img_size] * len(test_df),
                ),
                total=len(test_df),
            )
        )
