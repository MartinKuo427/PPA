import torch
from transformers import (
    set_seed,
)
import transformers
import torch
import numpy as np
import datasets
import argparse

transformers.logging.set_verbosity_error()


set_seed(0)  # for reproducibility
# Set the random seed for NumPy
np.random.seed(0)
# Set the random seed for PyTorch
torch.manual_seed(0)
# If you are using CUDA (i.e., a GPU), also set the seed for it
torch.cuda.manual_seed_all(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--privacy_data_path", type=str, default="", help="privacy_data_path")
    parser.add_argument("--name_prefix", type=str, default="", help="name_prefix")
    args = parser.parse_args()

    privacy_data_path = args.privacy_data_path
    name_prefix = args.name_prefix

    all_dataset = datasets.load_from_disk(privacy_data_path)

    print("all_dataset")
    print(all_dataset)

    total_length = len(all_dataset)
    train_length = int(total_length * 0.9)
    valid_length = total_length - train_length
    train_dataset = all_dataset.select(range(train_length))
    valid_dataset = all_dataset.select(range(train_length, total_length, 1))

    print("train_dataset")
    print(train_dataset)

    print("valid_dataset")
    print(valid_dataset)

    train_data_path = privacy_data_path + "_train"
    train_dataset.save_to_disk(train_data_path)

    valid_data_path = privacy_data_path + "_valid"
    valid_dataset.save_to_disk(valid_data_path)