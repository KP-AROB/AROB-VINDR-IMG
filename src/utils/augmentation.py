import os
import cv2
import logging
import albumentations as A
from glob import glob
from tqdm import tqdm


def augment_image(image_path, n_augment, pipeline):
    image = cv2.imread(image_path)
    augmented_images = []
    for _ in range(n_augment):
        augmented = pipeline(image=image)['image']
        augmented_images.append(augmented)
    return augmented_images


def make_classwise_augmentations(data_dir, n_augment, class_list=['mass', 'suspicious_calcification']):
    logging.info("Running data augmentation")

    augmentation_pipeline = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=45, border_mode=cv2.BORDER_CONSTANT, p=0.3),
        A.ElasticTransform(alpha=1, sigma=50, p=0.3),
        A.RandomBrightnessContrast(
            brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
    ])

    class_dirs = [os.path.join(data_dir, i) for i in class_list]
    for cls_path in class_dirs:
        number_of_images = glob(os.path.join(cls_path, '*.png'))
        with tqdm(total=len(number_of_images), desc=f"Augmenting {cls_path}") as pbar:
            for idx, img in enumerate(number_of_images):
                augmented_images = augment_image(
                    img, n_augment, augmentation_pipeline)
                for j, augmented_image in enumerate(augmented_images):
                    output_path = os.path.join(cls_path, f"aug_{idx}_{j}.png")
                    cv2.imwrite(output_path, augmented_image)
                pbar.update()

    logging.info("Augmentations finished.")
