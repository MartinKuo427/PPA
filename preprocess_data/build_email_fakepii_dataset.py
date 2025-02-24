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

def save_huggingace_dataset(global_name_twin_email_list, global_category_twin_email_list, global_twin_email_template_list, global_twin_email_template_type_list, 
                            global_twin_email_template_target_list, global_twin_email_target_pii_type_list):


    df = pd.DataFrame({
        'NAME': global_name_twin_email_list,
        'CATEGORY': global_category_twin_email_list,
        'TEMPLATE': global_twin_email_template_list,
        'TEMPLATE_TYPE': global_twin_email_template_type_list,
        'TARGET': global_twin_email_template_target_list,
        'TARGET_PII_TYPE': global_twin_email_target_pii_type_list
    }, columns=['NAME', 'CATEGORY', 'TEMPLATE', 'TEMPLATE_TYPE', 'TARGET', 'TARGET_PII_TYPE'])

    df_data = datasets.Dataset.from_pandas(df)

    directory_path = os.path.join("../main_code", "propile_fake_template_dataset")
    if not os.path.exists(directory_path):
        os.mkdir(directory_path)

    data_path = os.path.join(directory_path, "fake_email_number_twin_email_template_dataset")

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

    global_name_twin_email_list = []
    global_category_twin_email_list = []
    global_twin_email_template_list = []
    global_twin_email_template_type_list = []
    global_twin_email_template_target_list = []
    global_twin_email_target_pii_type_list = []

    email_target_pii_type = "email address"


    data_generator = PresidioDataGenerator()

    # first focus on pii
    for person_index in range(len(privacy_ground_truth_table)):

        gt_NAME = privacy_ground_truth_table['NAME'][person_index]

        category = privacy_ground_truth_table['CATEGORY'][person_index]

        gt_EMAIL_list = privacy_ground_truth_table['EMAIL'][person_index]
        gt_EMAIL_list = ast.literal_eval(gt_EMAIL_list)

        if (len(gt_EMAIL_list) > 0):
            template_type = "twin_email"
            target_pii_type = email_target_pii_type

            faked_email_templates = ["{{email}}"]
            fake_records = data_generator.generate_fake_data(
                templates=faked_email_templates, n_samples=1
            )

            fake_records = list(fake_records)
            fake_pii = fake_records[0].fake

            email_source_string = gt_NAME + " " + email_target_pii_type

            global_name_twin_email_list.append(gt_NAME)
            global_category_twin_email_list.append(category)
            global_twin_email_template_list.append(email_source_string)
            global_twin_email_template_type_list.append(template_type)
            global_twin_email_template_target_list.append(fake_pii)
            global_twin_email_target_pii_type_list.append(target_pii_type)

    save_huggingace_dataset(global_name_twin_email_list, global_category_twin_email_list, global_twin_email_template_list, global_twin_email_template_type_list, 
                            global_twin_email_template_target_list, global_twin_email_target_pii_type_list)