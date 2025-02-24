import torch
from transformers import (
    set_seed,
    AutoTokenizer,
)
import boto3
import transformers
import torch
import numpy as np
import os
from datasets import load_dataset
import pandas as pd
import datasets
from typing import Any, Dict, List
import contextlib
import argparse
import ast


transformers.logging.set_verbosity_error()


set_seed(0)  # for reproducibility

# Set the random seed for NumPy
np.random.seed(0)

# Set the random seed for PyTorch
torch.manual_seed(0)

# If you are using CUDA (i.e., a GPU), also set the seed for it
torch.cuda.manual_seed_all(0)


@contextlib.contextmanager
def main_process_first():
    if torch.distributed.is_initialized():
        is_main_process = torch.distributed.get_rank() == 0
        try:
            if not is_main_process:
                # tell all replicas to wait
                torch.distributed.barrier()
            yield
        finally:
            if is_main_process:
                torch.distributed.barrier()
    else:
        yield

def get_num_workers():
    # original
    # return min(torch.get_num_threads() // max(1, torch.cuda.device_count()), 32)
    # martinc test check
    return 1


def preprocess_dataset(
    dataset,
    tokenizer
):

    column_names = list(next(iter(dataset)).keys())

    def preprocess_pretrain_dataset(examples: Dict[str, List[Any]]) -> Dict[str, Any]:
        # build grouped texts with format `X1 X2 X3 ...`
        tokenized_examples = tokenizer(examples["TEXT"])
        return tokenized_examples

    preprocess_func = preprocess_pretrain_dataset

    with main_process_first():
        kwargs = {}
        num_threads = get_num_workers()

        kwargs = dict(
            num_proc=num_threads if num_threads > 0 else None,
            load_from_cache_file=False,
            desc="Running tokenizer on dataset"
        )

        dataset = dataset.map(
            preprocess_func,
            batched=True,            
            remove_columns=column_names,
            **kwargs
        )

    return dataset


def save_huggingace_dataset(global_name_twin_phone_list, global_category_twin_phone_list, global_twin_phone_template_list, global_twin_phone_template_type_list, global_twin_phone_template_target_list, global_twin_phone_target_pii_type_list,
                        global_name_twin_address_list, global_category_twin_address_list, global_twin_address_template_list, global_twin_address_template_type_list, global_twin_address_template_target_list, global_twin_address_target_pii_type_list):

    df = pd.DataFrame({
        'NAME': global_name_twin_phone_list,
        'CATEGORY': global_category_twin_phone_list,
        'TEMPLATE': global_twin_phone_template_list,
        'TEMPLATE_TYPE': global_twin_phone_template_type_list,
        'TARGET': global_twin_phone_template_target_list,
        'TARGET_PII_TYPE': global_twin_phone_target_pii_type_list
    }, columns=['NAME', 'CATEGORY', 'TEMPLATE', 'TEMPLATE_TYPE', 'TARGET', 'TARGET_PII_TYPE'])


    df_data = datasets.Dataset.from_pandas(df)

    directory_path = os.path.join("../main_code", "propile_dummy_template_dataset")
    if not os.path.exists(directory_path):
        os.mkdir(directory_path)

    data_path = os.path.join(directory_path, "dummy_phone_number_twin_phone_template_dataset")

    df_data.save_to_disk(data_path)


    # whalley_twin_phone_template_dataset
    df = pd.DataFrame({
        'NAME': global_name_twin_address_list,
        'CATEGORY': global_category_twin_address_list,
        'TEMPLATE': global_twin_address_template_list,
        'TEMPLATE_TYPE': global_twin_address_template_type_list,
        'TARGET': global_twin_address_template_target_list,
        'TARGET_PII_TYPE': global_twin_address_target_pii_type_list
    }, columns=['NAME', 'CATEGORY', 'TEMPLATE', 'TEMPLATE_TYPE', 'TARGET', 'TARGET_PII_TYPE'])

    df_data = datasets.Dataset.from_pandas(df)

    directory_path = os.path.join("../main_code", "propile_dummy_template_dataset")
    if not os.path.exists(directory_path):
        os.mkdir(directory_path)

    data_path = os.path.join(directory_path, "dummy_address_twin_address_template_dataset")

    df_data.save_to_disk(data_path)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--privacy_data_path", type=str, default="", help="rootdir")
    parser.add_argument("--tokenizer_name", type=str, default="meta-llama/Llama-2-7b", help="tokenizer_name")

    args = parser.parse_args()

    privacy_data_path = args.privacy_data_path
    tokenizer_name = args.tokenizer_name


    # Load LLaMA tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training

    privacy_ground_truth_table = load_dataset("csv", data_files=privacy_data_path)['train']

    global_name_twin_phone_list = []
    global_category_twin_phone_list = []
    global_twin_phone_template_list = []
    global_twin_phone_template_type_list = []
    global_twin_phone_template_target_list = []
    global_twin_phone_target_pii_type_list = []

    global_name_twin_address_list = []
    global_category_twin_address_list = []
    global_twin_address_template_list = []
    global_twin_address_template_type_list = []
    global_twin_address_template_target_list = []
    global_twin_address_target_pii_type_list = []

    phone_target_pii_type = "phone number"
    address_target_pii_type = "address"



    # first focus on pii
    for person_index in range(len(privacy_ground_truth_table)):

        gt_NAME = privacy_ground_truth_table['NAME'][person_index]

        category = privacy_ground_truth_table['CATEGORY'][person_index]

        gt_PHONE_list = privacy_ground_truth_table['PHONE'][person_index]
        gt_PHONE_list = ast.literal_eval(gt_PHONE_list)

        gt_ADDRESS_list = privacy_ground_truth_table['ADDRESS_ORI'][person_index]
        gt_ADDRESS_list = ast.literal_eval(gt_ADDRESS_list)

        if (len(gt_PHONE_list) > 0):
            template_type = "twin_phone"
            target_pii_type = phone_target_pii_type

            fake_pii = "dummy"

            phone_source_string = gt_NAME + " " + phone_target_pii_type

            global_name_twin_phone_list.append(gt_NAME)
            global_category_twin_phone_list.append(category)
            global_twin_phone_template_list.append(phone_source_string)
            global_twin_phone_template_type_list.append(template_type)
            global_twin_phone_template_target_list.append(fake_pii)
            global_twin_phone_target_pii_type_list.append(target_pii_type)
        
        if (len(gt_ADDRESS_list) > 0):
            template_type = "twin_address"
            target_pii_type = address_target_pii_type

            fake_pii = "dummy"

            address_source_string = gt_NAME + " " + address_target_pii_type

            global_name_twin_address_list.append(gt_NAME)
            global_category_twin_address_list.append(category)
            global_twin_address_template_list.append(address_source_string)
            global_twin_address_template_type_list.append(template_type)
            global_twin_address_template_target_list.append(fake_pii)
            global_twin_address_target_pii_type_list.append(target_pii_type)

    save_huggingace_dataset(global_name_twin_phone_list, global_category_twin_phone_list, global_twin_phone_template_list, global_twin_phone_template_type_list, global_twin_phone_template_target_list, global_twin_phone_target_pii_type_list,
                            global_name_twin_address_list, global_category_twin_address_list, global_twin_address_template_list, global_twin_address_template_type_list, global_twin_address_template_target_list, global_twin_address_target_pii_type_list)