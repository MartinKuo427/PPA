import torch
from transformers import (
    AutoTokenizer,
    set_seed,
)
import boto3
import transformers
import torch
import numpy as np
import os
from datasets import load_dataset
import pandas as pd
from typing import Any, Dict, List
from itertools import chain
import contextlib
import ast
import argparse

transformers.logging.set_verbosity_error()


set_seed(0)  # for reproducibility

# Set the random seed for NumPy
np.random.seed(0)

# Set the random seed for PyTorch
torch.manual_seed(0)

# If you are using CUDA (i.e., a GPU), also set the seed for it
torch.cuda.manual_seed_all(0)

# aws service load
client = boto3.client('comprehend')

class Client:
    def __init__(self):
        self.CATEGORY = ""
        self.template_type = []
        self.template = []
        self.template_output = []
        self.aws_comprehend_output = []
        self.EXPOSE_ADDRESS_ITEM = []
        self.EXPOSE_PHONE_ITEM = []

        self.database_dict  = {
            'expose_item': {
                'ADDRESS': self.EXPOSE_ADDRESS_ITEM,
                'PHONE': self.EXPOSE_PHONE_ITEM
            }
        }        

    def add_PII(self, element, type, predict_pii_name):
        if (len(self.database_dict[type][predict_pii_name]) == 0):
            self.database_dict[type][predict_pii_name] = element
        else:
            self.database_dict[type][predict_pii_name].append(element)


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
        return min(torch.get_num_threads() // max(1, torch.cuda.device_count()), 32)

def preprocess_dataset(
    dataset,
    tokenizer
):

    column_names = list(next(iter(dataset)).keys())

    def preprocess_pretrain_dataset(examples: Dict[str, List[Any]]) -> Dict[str, Any]:
        # build grouped texts with format `X1 X2 X3 ...`
        kwargs = dict(add_special_tokens=True)

        if hasattr(tokenizer, "add_eos_token"): # for LLaMA tokenizer
            setattr(tokenizer, "add_eos_token", True)

        # tokenized_examples = tokenizer(examples["prompt"], **kwargs)
        tokenized_examples = tokenizer(examples["TEXT"], **kwargs)
        concatenated_examples = {k: list(chain(*tokenized_examples[k])) for k in tokenized_examples.keys()}
        total_length = len(concatenated_examples[list(concatenated_examples.keys())[0]])

        cutoff_len = 128
        block_size = cutoff_len

        # we drop the small remainder, and if the total_length < block_size, we exclude this batch
        total_length = (total_length // block_size) * block_size

        # martinc add function
        if (total_length == 0):
            total_length = block_size

        # split by chunks of cutoff_len
        result = {
            k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        return result

    dataset = dataset.filter(lambda example: example["TEXT"])
    preprocess_func = preprocess_pretrain_dataset

    with main_process_first():
        kwargs = {}

        num_threads = get_num_workers()

        kwargs = dict(
            num_proc=num_threads if num_threads > 0 else None,
            desc="Running tokenizer on dataset"
        )

        dataset = dataset.map(
            preprocess_func,
            batched=True,            
            remove_columns=column_names,
            **kwargs
        )
        
    return dataset

def save_record_to_excel(output_filepath, person_pii_dict, privacy_ground_truth_table):
    NAME_list = []
    CATEGORY_list = []
    TEMPLATE_TYPE = []
    TEMPLATE = []
    TEMPLATE_OUTPUT = []
    AWS_COMPREHEND_OUTPUT = []
    EXPOSE_ADDRESS_ITEM = []
    EXPOSE_PHONE_ITEM = []


    for person_index in range(len(privacy_ground_truth_table)):
        gt_NAME = privacy_ground_truth_table['NAME'][person_index]
        person_pii = person_pii_dict[gt_NAME]

        category = person_pii.CATEGORY

        NAME_list.append(gt_NAME)
        CATEGORY_list.append(category)

        template_type = person_pii.template_type
        template = person_pii.template
        template_output = person_pii.template_output
        aws_comprehend_output = person_pii.aws_comprehend_output

        TEMPLATE_TYPE.append(template_type)
        TEMPLATE.append(template)
        TEMPLATE_OUTPUT.append(template_output)
        AWS_COMPREHEND_OUTPUT.append(aws_comprehend_output)

        item = person_pii.database_dict["expose_item"]["ADDRESS"] if len(person_pii.database_dict["expose_item"]["ADDRESS"]) > 0 else "-"
        EXPOSE_ADDRESS_ITEM.append(item)
        item = person_pii.database_dict["expose_item"]["PHONE"] if len(person_pii.database_dict["expose_item"]["PHONE"]) > 0 else "-"
        EXPOSE_PHONE_ITEM.append(item)

    # Create a DataFrame
    df = pd.DataFrame({
        'NAME': NAME_list,
        'CATEGORY': CATEGORY_list,
        'TEMPLATE_TYPE': TEMPLATE_TYPE,
        'TEMPLATE': TEMPLATE,
        'TEMPLATE_OUTPUT': TEMPLATE_OUTPUT,
        'AWS_COMPREHEND_OUTPUT': AWS_COMPREHEND_OUTPUT,
        'EXPOSE_ADDRESS_ITEM': EXPOSE_ADDRESS_ITEM,
        'EXPOSE_PHONE_ITEM': EXPOSE_PHONE_ITEM
    }, columns=['NAME', 'CATEGORY', 'TEMPLATE_TYPE', 'TEMPLATE', 'TEMPLATE_OUTPUT', 'AWS_COMPREHEND_OUTPUT', 'EXPOSE_ADDRESS_ITEM', 'EXPOSE_PHONE_ITEM'])

    df.to_csv(output_filepath)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--rootdir", type=str, default="", help="rootdir")
    parser.add_argument("--privacy_data_path", type=str, default="../aeslc_train_privacy_ground_truth.csv", help="privacy_data_path")
    parser.add_argument("--tokenizer_name", type=str, default="meta-llama/Llama-2-7b-hf", help="tokenizer_name")
    parser.add_argument("--output_filepath", type=str, default="./data/", help="output directory")

    args = parser.parse_args()

    rootdir = args.rootdir
    privacy_data_path = args.privacy_data_path
    output_filepath = args.output_filepath

    tokenizer_name = args.tokenizer_name


    privacy_ground_truth_table = load_dataset("csv", data_files=privacy_data_path)['train']

    print("privacy_ground_truth_table")
    print(privacy_ground_truth_table)


    person_pii_dict = {} # Client

    for person_index in range(len(privacy_ground_truth_table)):
        new_client = Client()

        gt_NAME = privacy_ground_truth_table['NAME'][person_index]

        gt_CATEGORY = privacy_ground_truth_table['CATEGORY'][person_index]
        new_client.CATEGORY = gt_CATEGORY

        person_pii_dict[gt_NAME] = new_client


    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training


    filnames_list = []

    for directory, subdirectory, filnames in os.walk(rootdir):
        for i in range(len(filnames)):
            path_string = directory + "/" + filnames[i]
            filnames_list.append(path_string)

    print("filnames_list")
    print(filnames_list)


    pii_detect_list = ["ADDRESS", "PHONE"]

    for filnames in filnames_list:
        part_dataset = load_dataset("csv", data_files=filnames)['train']

        for d_index in range(len(part_dataset)):
            name = part_dataset['NAME'][d_index]
            category = part_dataset['CATEGORY'][d_index]
            template_type = part_dataset['TEMPLATE_TYPE'][d_index]

            template = part_dataset['TEMPLATE'][d_index]
            template_output = part_dataset['TEMPLATE_OUTPUT'][d_index]
            aws_comprehend_output = part_dataset['AWS_COMPREHEND_OUTPUT'][d_index]

            template = ast.literal_eval(template)
            template_output = ast.literal_eval(template_output)
            aws_comprehend_output = ast.literal_eval(aws_comprehend_output)

            client = person_pii_dict[name]

            client.template_type.append(template_type)
            client.template.append(template)

            real_template_output = []
            # template_output is included in template, we need to extract real template_output from template
            for t_index in range(len(template)):
                temp_template = template[t_index]
                temp_template_output = template_output[t_index]
                temp_template_length = len(temp_template)
                temp_real_template_output = temp_template_output[temp_template_length:]
                real_template_output.append(temp_real_template_output)

            client.template_output.append(real_template_output)
            client.aws_comprehend_output.append(aws_comprehend_output)

            for aws_index in range(len(aws_comprehend_output)):
                temp_template_output = template_output[aws_index]
                detect_pii_response_list = aws_comprehend_output[aws_index]
                for detect_index in range(len(detect_pii_response_list)):
                    
                    detect_pii_response = detect_pii_response_list[detect_index]
                    detect_pii_type = detect_pii_response['Type']

                    if (detect_pii_type in pii_detect_list):
                        BeginOffset = detect_pii_response['BeginOffset']
                        EndOffset = detect_pii_response['EndOffset']
                        pii_item = temp_template_output[BeginOffset:EndOffset]
                        if (template_type == "twin_address"):
                            if (pii_item not in client.database_dict["expose_item"][detect_pii_type]):
                                client.database_dict["expose_item"][detect_pii_type].append(pii_item)

    save_record_to_excel(output_filepath, person_pii_dict, privacy_ground_truth_table)