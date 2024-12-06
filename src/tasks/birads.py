import os
import cv2
import logging
from tqdm import tqdm
from src.utils.image import get_final_image
from concurrent.futures import ProcessPoolExecutor
from src.utils.dataframe import prepare_vindr_birads_dataframe


def prepare_row(row, data_dir: str, out_dir: str, img_size: int):
    try:
        sample_path = os.path.join(
            data_dir, 'images', row['study_id'], row['image_id'] + '.dicom')
        image = get_final_image(sample_path, img_size)
        output_image_path = os.path.join(
            out_dir, row['breast_birads'], "{}.png".format(row.name))
        cv2.imwrite(output_image_path, image)
    except Exception as e:
        img_id = row['image_id']
        logging.error(f'Failed to process image {img_id}: {e}')


def prepare_birads_dataset(data_dir: str, out_dir: str, img_size: int):
    """Prepare the VINDR MAMMO dataset for birads specific classification
    Args:
        data_dir (str): Path to original cbis dataset
        out_dir (str): Path to save the prepared cbis dataset
        img_size (int): New image size
    """
    train_df = prepare_vindr_birads_dataframe(data_dir, True)
    test_df = prepare_vindr_birads_dataframe(data_dir, False)

    for i in train_df['breast_birads'].unique():
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
