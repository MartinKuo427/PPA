import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
)
import boto3
import transformers
import torch
import numpy as np
import pandas as pd
import datasets
import contextlib
from tqdm import tqdm
import os
import argparse

transformers.logging.set_verbosity_error()

set_seed(0)  # for reproducibility
# Set the random seed for NumPy
np.random.seed(0)
# Set the random seed for PyTorch
torch.manual_seed(0)
# If you are using CUDA (i.e., a GPU), also set the seed for it
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True

# aws service load
client = boto3.client('comprehend')

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


def save_propile_record_rank_to_excel(output_filepath, start_data_point, end_data_point, global_name_list, global_category_list, global_template_type_list, global_template_input_list, global_template_output_list, global_aws_comprehend_output_list):
    # Create a DataFrame
    df = pd.DataFrame({
        'NAME': global_name_list,
        'CATEGORY': global_category_list,
        'TEMPLATE_TYPE': global_template_type_list,
        'TEMPLATE': global_template_input_list,
        'TEMPLATE_OUTPUT': global_template_output_list,
        'AWS_COMPREHEND_OUTPUT': global_aws_comprehend_output_list
    }, columns=['NAME', 'CATEGORY', 'TEMPLATE_TYPE', 'TEMPLATE', 'TEMPLATE_OUTPUT', 'AWS_COMPREHEND_OUTPUT'])    
    """
    if not os.path.exists(output_filepath):
        os.mkdir(output_filepath)
    """
    os.makedirs(output_filepath, exist_ok=True)
    output_filepath = os.path.join(output_filepath, "_s_" + str(start_data_point) + "_e_" + str(end_data_point) + "_.csv")
    # output_filepath = output_filepath + "_s_" + str(start_data_point) + "_e_" + str(end_data_point) + "_.csv"

    print("temp save output_filepath:", output_filepath)
    df.to_csv(output_filepath)

    print("save_record_rank_to_excel output_filepath:", output_filepath)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_data_point", type=int, default=0)
    parser.add_argument("--end_data_point", type=int, default=2484)
    parser.add_argument("--attack_dataset_path", type=str, default="", help="attack_dataset_path")
    parser.add_argument("--tokenizer_name", type=str, default="meta-llama/Llama-2-7b-hf", help="tokenizer_name")
    parser.add_argument("--model_name", type=str, default="", help="model_name")
    parser.add_argument("--output_filepath", type=str, default="", help="output directory")

    args = parser.parse_args()

    attack_dataset_path = args.attack_dataset_path
    tokenizer_name = args.tokenizer_name
    model_name = args.model_name
    print("model_name:", model_name)
    output_filepath = args.output_filepath
    print("output_filepath:", output_filepath)
    start_data_point = args.start_data_point
    


    train_users_twin_phone_template_dataset = datasets.load_from_disk(attack_dataset_path)

    whole_template_dataset = train_users_twin_phone_template_dataset

    PII_length_dict = {
        'email address': 20
    }

    # Load LLaMA tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto"
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1


    global_name_list = []
    global_category_list = []
    global_template_type_list = []
    global_template_input_list = []
    global_template_output_list = []
    global_aws_comprehend_output_list = []

    end_data_point = len(whole_template_dataset)

    for index in tqdm(range(start_data_point, end_data_point, 1)):
        gt_NAME = whole_template_dataset['NAME'][index]
        category = whole_template_dataset['CATEGORY'][index]
        template_type = whole_template_dataset['TEMPLATE_TYPE'][index]
        template_input = whole_template_dataset['TEMPLATE'][index]
        target_pii_type = whole_template_dataset['TARGET_PII_TYPE'][index]

        pii_length = PII_length_dict[target_pii_type]
        
        model_input = tokenizer(template_input, return_tensors="pt", padding=True, truncation=True).to(device='cuda')

        model.eval()

        with torch.no_grad():
            template_output_ids = model.generate(**model_input, max_new_tokens=pii_length, pad_token_id=tokenizer.pad_token_id, num_beams=3, num_return_sequences=1, early_stopping=True, max_time=600.0)

            template_output_batch = tokenizer.batch_decode(template_output_ids, skip_special_tokens=True)

            global_name_list.append(gt_NAME)
            global_category_list.append(category)
            global_template_type_list.append(template_type)
            global_template_input_list.append(template_input)
            global_template_output_list.append(template_output_batch)

            temp_aws_comprehend_output_list = []

            for output_index in range(len(template_output_batch)):
                output_predict_str = template_output_batch[output_index]
                output_pii_response_list_e = client.detect_pii_entities(
                    Text=output_predict_str,
                    LanguageCode='en'
                )

                detect_pii_response_list = output_pii_response_list_e['Entities']
                temp_aws_comprehend_output_list.append(detect_pii_response_list)
            
            global_aws_comprehend_output_list.append(temp_aws_comprehend_output_list)

    save_propile_record_rank_to_excel(output_filepath, start_data_point, end_data_point, global_name_list, global_category_list, global_template_type_list, global_template_input_list, global_template_output_list, global_aws_comprehend_output_list)