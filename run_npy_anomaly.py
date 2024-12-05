from glob import glob
from tqdm import tqdm, trange
import numpy as np
import cv2
import os

train_folder = '/mnt/d/datasets/vindr-mammo/data/npy/train'
test_folder = '/mnt/d/datasets/vindr-mammo/data/npy/test'
out_folder = '/mnt/d/datasets/vindr-mammo/data/raw_anomaly'


train_image_files = glob(train_folder + '/*images*.npy')
train_label_files = glob(train_folder + '/*labels*.npy')

test_image_files = glob(test_folder + '/*images*.npy')
test_label_files = glob(test_folder + '/*labels*.npy')

classes = ['no_finding', 'suspicious_calcification',
           'mass']

os.makedirs(os.path.join(out_folder, 'train', 'normal'), exist_ok=True)
os.makedirs(os.path.join(out_folder, 'train', 'abnormal'), exist_ok=True)
os.makedirs(os.path.join(out_folder, 'test', 'normal'), exist_ok=True)
os.makedirs(os.path.join(out_folder, 'test', 'abnormal'), exist_ok=True)

with tqdm(total=len(train_image_files), desc='Saving training images') as pbar_outer:
    for idx, (image_file, label_file) in enumerate(zip(train_image_files, train_label_files)):
        img_tensor = np.load(image_file)
        label_tensor = np.load(label_file)
        with tqdm(total=len(img_tensor), desc=f'Processing', leave=False) as pbar_inner:
            for jdx, (img, label) in enumerate(zip(img_tensor, label_tensor)):
                img = cv2.normalize(img, None, 0, 255,
                                    cv2.NORM_MINMAX).astype('uint8')

                new_class = 'normal' if classes[label] == 'no_finding' else 'abnormal'

                cv2.imwrite(os.path.join(
                    out_folder, 'train', new_class, f'{idx}_{jdx}.png'), img[0])
                pbar_inner.update(1)
        pbar_outer.update(1)


with tqdm(total=len(test_image_files), desc='Saving test images') as pbar_outer:
    for idx, (image_file, label_file) in enumerate(zip(test_image_files, test_label_files)):
        img_tensor = np.load(image_file)
        label_tensor = np.load(label_file)
        with tqdm(total=len(img_tensor), desc=f'Processing', leave=False) as pbar_inner:
            for jdx, (img, label) in enumerate(zip(img_tensor, label_tensor)):
                img = cv2.normalize(img, None, 0, 255,
                                    cv2.NORM_MINMAX).astype('uint8')

                new_class = 'normal' if classes[label] == 'no_finding' else 'abnormal'
                cv2.imwrite(os.path.join(
                    out_folder, 'test', new_class, f'{idx}_{jdx}.png'), img[0])
                pbar_inner.update(1)
        pbar_outer.update(1)
