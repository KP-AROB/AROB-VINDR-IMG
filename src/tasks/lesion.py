import os
import cv2
import shutil
from tqdm import tqdm
from glob import glob
from src.utils.image import load_dicom_image
from concurrent.futures import ProcessPoolExecutor
from src.utils.dataframe import prepare_vindr_dataframe


def prepare_row(row, data_dir: str, out_dir: str, img_size: int):
    sample_path = os.path.join(
        data_dir, 'images', row['study_id'], row['image_id'] + '.dicom')
    original_image = load_dicom_image(sample_path)
    resized_image = cv2.resize(
        original_image,
        (img_size, img_size),
        interpolation=cv2.INTER_LINEAR,
    )
    output_image_path = os.path.join(
        out_dir, row['finding_categories'], "{}.png".format(row.name))
    cv2.imwrite(output_image_path, resized_image)
    return


def prepare_lesion_dataset(data_dir: str, out_dir: str, img_size: int):
    """Prepare the VINDR MAMMO dataset for lesion specific classification

    Args:
        data_dir (str): Path to original cbis dataset
        out_dir (str): Path to save the prepared cbis dataset
        img_size (int): New image size
        severity (bool): Whether to create classes for pathologies or not
    """
    shutil.rmtree(os.path.join(out_dir), ignore_errors=True)
    class_list = ['no_finding', 'suspicious_calcification', 'mass']
    train_df = prepare_vindr_dataframe(data_dir, class_list, True)
    test_df = prepare_vindr_dataframe(data_dir, class_list, False)

    for i in class_list:
        os.makedirs(os.path.join(out_dir, 'train', i), exist_ok=True)
        os.makedirs(os.path.join(out_dir, 'test', i), exist_ok=True)

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
