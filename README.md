# Vindr-mammo dataset preparation


This code converts the Vindr-Mammo dataset into grouped images to use in an ImageFolder PyTorch Dataset.
So far, the labels contained in the prepared dataset are : 

- no_finding
- suspicious_calcification
- mass


## 1. Requirements

```bash
pip install -r requirements.txt
```

## 2. Installation

First, you need to download the dataset. It is provided by the authors [Here](https://physionet.org/content/vindr-mammo/)
Then just clone this repository.

## 3. Usage

You can run the preparation script with the following command and given flags : 

```bash
python run.py --data_dir ./cbis_ddsm --out_dir ./data
```

| Flag                  | Description                                                                                                       | Default Value   |
|-----------------------|-------------------------------------------------------------------------------------------------------------------|-----------------|
| --data_dir            | The folder where the Vindr-Mammo dataset is stored                                                                | ./data          |
| --out_dir             | The folder where the prepared dataset will be stored                                                              | ./data          |
| --img_size            | The size to which the image should be resized                                                                     | 224             |
| --n_augment           | The number of new images to create with augmentations                                                             | 0               |
| --augment_type        | The type of augmentation to perform ('photometric' or 'geometric')                                                | 'photometric    |
| --task                | The task for which the dataset will be prepared                                                                   | 'birads'        |

### 3.1. Dataset task

We implemented different ways to prepare the dataset depending on the targetted classification system development. 

- ```birads```: It prepares the dataset with classes following the 3-category birads system (0 - recall, 1 - normal, 2 - benign).
- ```lesions```: It prepares the dataset with classes corresponding to the three default classes (```no_finding```, ```suspicious_calcification```, ```mass```).
- ```anomaly```: It prepares the dataset for a binary classification task by grouping anomaly classes together.

### 3.2. Preprocessing

For each task the images are loaded and normalized using the truncated normalization method.
This step is done by first cropping the image to the breast region through the Otsu threshold method.

### 3.3. File structure

For each task the script will create training and testing sets based on the original dataset split. For a given task the file structure will then look like :

- 📂 data/
    - 📂 task_name/
        - 📂 train/
            - 📂 calc/
                - 📄 01.png
                - 📄 02.png
            - 📂 mass/
                - 📄 01.png
                - 📄 02.png
        - 📂 test/
            - 📂 mass/
                - 📄 01.png
                - 📄 02.png
            - 📂 calc/
                - 📄 01.png
                - 📄 02.png


### 3.4. Data augmentation

The dataset can be augmented during the preparation process following a pre-defined pipeline, the augmentation can be called with the ```--aug_ratio``` flag.
This ratio controls the amount of new images (per scan) that will be created. By default this flag has a value of 0 meaning that the dataset will not be augmented.

This augmentation process uses the albumentations library and the augmentation pipeline follows the following code : 

```python

transform = A.Compose([
     A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=45, always_apply=True,
                border_mode=cv2.BORDER_CONSTANT, p=1.0),
])

```

## 4. Data Statistics

- ./data/scan-severity/train - Mean: 0.2095540165901184, Std: 0.2696904242038727
- ./data/roi-severity/train - Mean: 0.3523887097835541, Std: 0.24902833998203278