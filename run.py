import argparse
import logging
import os
import shutil
from src.tasks.lesion import prepare_lesion_dataset
from src.tasks.anomaly import prepare_anomaly_dataset
from src.tasks.birads import prepare_birads_dataset
from src.utils.augmentation import make_classwise_augmentations

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Vindr Mammo image dataset preparation")
    parser.add_argument("--data_dir", type=str, default='./data')
    parser.add_argument("--out_dir", type=str, default='./data')
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--n_augment", type=int, default=0)
    parser.add_argument("--augment_type", type=str,
                        default='photometric',  choices=['geometric', 'photometric'])
    parser.add_argument("--task", type=str, default='birads',
                        choices=['anomalies', 'lesions', 'birads'])

    args = parser.parse_args()
    parser.set_defaults(synthetize=False)

    # INIT
    logging_message = "[AROB-2025-KAPTIOS-VINDR-IMG]"

    logging.basicConfig(
        level=logging.INFO,
        format=f'%(asctime)s - {logging_message} - %(levelname)s - %(message)s'
    )
    logging.info('Running Vindr Mammo image dataset preparation')

    data_dir = args.data_dir
    out_dir = args.out_dir

    if out_dir != args.data_dir:
        shutil.rmtree(out_dir)

    train_folder = os.path.join(out_dir, 'train')
    test_folder = os.path.join(out_dir, 'test')

    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    class_list = ['no_finding', 'suspicious_calcification',
                  'mass']

    # # PREPARATION
    if args.task == 'anomalies':
        prepare_anomaly_dataset(data_dir, out_dir,
                                args.img_size, class_list)
    elif args.task == 'lesions':
        prepare_lesion_dataset(data_dir, out_dir,
                               args.img_size, class_list)
    elif args.task == 'birads':
        prepare_birads_dataset(data_dir, out_dir,
                               args.img_size)

    if args.n_augment > 0:
        make_classwise_augmentations(
            train_folder, args.n_augment, class_list[1:], args.augment_type)
