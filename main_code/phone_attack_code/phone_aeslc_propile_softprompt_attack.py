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
from peft import get_peft_model, TaskType, PromptTuningConfig, PromptTuningInit
from tqdm import tqdm
from safetensors import safe_open
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


    os.makedirs(output_filepath, exist_ok=True)
    output_filepath = os.path.join(output_filepath, "_s_" + str(start_data_point) + "_e_" + str(end_data_point) + "_.csv")
    # output_filepath = output_filepath + "_s_" + str(start_data_point) + "_e_" + str(end_data_point) + "_.csv"

    df.to_csv(output_filepath)

    print("save_record_rank_to_excel output_filepath:", output_filepath)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_data_point", type=int, default=0)
    parser.add_argument("--end_data_point", type=int, default=577)
    parser.add_argument("--attack_dataset_path", type=str, default="", help="attack_dataset_path")
    parser.add_argument("--tokenizer_name", type=str, default="meta-llama/Llama-2-7b-hf", help="tokenizer_name")
    parser.add_argument("--model_name", type=str, default="", help="model_name")
    parser.add_argument("--peft_model_id", type=str, default="", help="peft_model_id")
    parser.add_argument("--num_virtual_tokens", type=int, default=20)
    parser.add_argument("--output_filepath", type=str, default="./data/", help="output directory")

    args = parser.parse_args()

    attack_dataset_path = args.attack_dataset_path
    print("attack_dataset_path:", attack_dataset_path)

    tokenizer_name = args.tokenizer_name
    print("tokenizer_name:", tokenizer_name)

    model_name = args.model_name

    print("model_name:", model_name)

    peft_model_id = args.peft_model_id
    print("peft_model_id:", peft_model_id)

    num_virtual_tokens = args.num_virtual_tokens
    print("num_virtual_tokens:", num_virtual_tokens)

    output_filepath = args.output_filepath
    print("output_filepath:", output_filepath)

    start_data_point = args.start_data_point
    end_data_point = args.end_data_point
    print("start_data_point:", start_data_point, " end_data_point:", end_data_point)


    train_users_twin_phone_template_dataset = datasets.load_from_disk(attack_dataset_path)

    whole_template_dataset = train_users_twin_phone_template_dataset

    print("whole_template_dataset")
    print(whole_template_dataset)

    PII_length_dict = {
        'phone number': 15
    }

    PII_length_key = list(PII_length_dict)


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

    # initialize pii_type Soft Prompt
    config = PromptTuningConfig(
        peft_type="PROMPT_TUNING",
        task_type=TaskType.CAUSAL_LM,
        prompt_tuning_init = PromptTuningInit.TEXT,
        prompt_tuning_init_text="phone",
        num_virtual_tokens=num_virtual_tokens,
        tokenizer_name_or_path = tokenizer_name)

    model = get_peft_model(model, config)

    safetensor_path = peft_model_id + "/adapter_model.safetensors"

    tensors = {}
    with safe_open(safetensor_path, framework="pt", device="cuda") as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)

    prompt_embeddings_tensors = tensors['prompt_embeddings']

    index = 0
    for key, backbone_layer in model.named_parameters():
        if (key == "prompt_encoder.default.embedding.weight"):
            backbone_layer.data.copy_(prompt_embeddings_tensors)
        index += 1

    global_name_list = []
    global_category_list = []
    global_template_type_list = []
    global_template_input_list = []
    global_template_output_list = []
    global_aws_comprehend_output_list = []

    for index in tqdm(range(start_data_point, end_data_point, 1)):
        gt_NAME = whole_template_dataset['NAME'][index]
        category = whole_template_dataset['CATEGORY'][index]
        template_type = whole_template_dataset['TEMPLATE_TYPE'][index]
        template_input = whole_template_dataset['TEMPLATE'][index]
        target_pii_type = whole_template_dataset['TARGET_PII_TYPE'][index]

        pii_length = PII_length_dict[target_pii_type]
        
        # print("start_data_point:", start_data_point, " end_data_point:", end_data_point, " index:", index, " gt_NAME:", gt_NAME)
        model_input = tokenizer(template_input, return_tensors="pt", padding=True, truncation=True).to(device='cuda')
        model.eval()

        model_kwargs = {"past_key_values": None}

        with torch.no_grad():
            template_output_ids = model.generate(**model_input, max_new_tokens=pii_length, pad_token_id=tokenizer.pad_token_id, num_beams=3, num_return_sequences=1, early_stopping=True, max_time=600.0, use_cache=True)# use_cache=True, **model_kwargs

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