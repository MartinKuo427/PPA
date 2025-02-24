import torch
from transformers import (
    AutoTokenizer,
    set_seed,
)
import transformers
import torch
import numpy as np
from datasets import load_dataset
import pandas as pd
import datasets
import argparse
import ast
import os
from presidio_evaluator.data_generator import PresidioDataGenerator

transformers.logging.set_verbosity_error()


set_seed(0)  # for reproducibility
# Set the random seed for NumPy
np.random.seed(0)
# Set the random seed for PyTorch
torch.manual_seed(0)
# If you are using CUDA (i.e., a GPU), also set the seed for it
torch.cuda.manual_seed_all(0)

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

    directory_path = os.path.join("../main_code", "propile_fake_template_dataset")
    if not os.path.exists(directory_path):
        os.mkdir(directory_path)

    data_path = os.path.join(directory_path, "fake_phone_number_twin_phone_template_dataset")

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

    directory_path = os.path.join("../main_code", "propile_fake_template_dataset")
    if not os.path.exists(directory_path):
        os.mkdir(directory_path)

    data_path = os.path.join(directory_path, "fake_address_twin_address_template_dataset")

    df_data.save_to_disk(data_path)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--privacy_data_path", type=str, default="", help="rootdir")
    parser.add_argument("--tokenizer_name", type=str, default="meta-llama/Llama-2-7b-hf", help="tokenizer_name")

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


    data_generator = PresidioDataGenerator()

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

            faked_phone_templates = ["{{phone_number}}"]
            fake_records = data_generator.generate_fake_data(
                templates=faked_phone_templates, n_samples=1
            )

            fake_records = list(fake_records)
            fake_pii = fake_records[0].fake

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

            faked_phone_templates = ["{{address}}"]
            fake_records = data_generator.generate_fake_data(
                templates=faked_phone_templates, n_samples=1
            )

            fake_records = list(fake_records)
            fake_pii = fake_records[0].fake

            address_source_string = gt_NAME + " " + address_target_pii_type

            global_name_twin_address_list.append(gt_NAME)
            global_category_twin_address_list.append(category)
            global_twin_address_template_list.append(address_source_string)
            global_twin_address_template_type_list.append(template_type)
            global_twin_address_template_target_list.append(fake_pii)
            global_twin_address_target_pii_type_list.append(target_pii_type)

    save_huggingace_dataset(global_name_twin_phone_list, global_category_twin_phone_list, global_twin_phone_template_list, global_twin_phone_template_type_list, global_twin_phone_template_target_list, global_twin_phone_target_pii_type_list,
                            global_name_twin_address_list, global_category_twin_address_list, global_twin_address_template_list, global_twin_address_template_type_list, global_twin_address_template_target_list, global_twin_address_target_pii_type_list)