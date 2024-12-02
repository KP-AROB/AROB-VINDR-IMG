import argparse
import logging
import os
from src.tasks.lesion import prepare_lesion_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Vindr Mammo image dataset preparation")
    parser.add_argument("--data_dir", type=str, required='./data')
    parser.add_argument("--out_dir", type=str, default='./data')
    parser.add_argument("--img_size", type=int, default=256)
    args = parser.parse_args()
    parser.set_defaults(synthetize=False)

    # INIT
    logging_message = "[AROB-2025-KAPTIOS-VINDR-IMG]"

    logging.basicConfig(
        level=logging.INFO,
        format=f'%(asctime)s - {logging_message} - %(levelname)s - %(message)s'
    )
    logging.info('Running Vindr Mammo image dataset preparation')

    out_dir = os.path.join(args.out_dir, 'ImageFolder')
    train_folder = os.path.join(out_dir, 'train')
    test_folder = os.path.join(out_dir, 'test')

    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    # PREPARATION
    prepare_lesion_dataset(args.data_dir, out_dir, args.img_size)
