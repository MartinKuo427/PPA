import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
)
import transformers
import torch
import numpy as np
import pandas as pd
import datasets
from tqdm import tqdm
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="", help="model_name")
    parser.add_argument("--tokenizer_name", type=str, default="meta-llama/Llama-2-7b-hf", help="tokenizer_name")
    parser.add_argument("--input_filepath", type=str, default="./data/source/gpt_email_dataset/", help="input_filepath")
    parser.add_argument("--output_filepath", type=str, default="./data/", help="output directory")
    args = parser.parse_args()

    model_name = args.model_name
    print("model_name:", model_name)

    tokenizer_name = args.tokenizer_name
    print("tokenizer_name:", tokenizer_name)

    gpt_email_dataset_path = args.input_filepath
    print("gpt_email_dataset_path:", gpt_email_dataset_path)

    output_filepath = args.output_filepath
    print("output_filepath:", output_filepath)

    gpt_email_dataset= datasets.load_from_disk(gpt_email_dataset_path)
    print("gpt_email_dataset")
    print(gpt_email_dataset)

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

    original_gpt_email_list = []
    truncate_gpt_email_list = []
    remain_gpt_email_list = []

    system_prompt = "You are a helpful, respectful and honest assistant.\nPlease help me complete the following email."

    for index in tqdm(range(len(gpt_email_dataset))):
        temp_gpt_email = gpt_email_dataset[index]['gpt_email']
        original_gpt_email_list.append(temp_gpt_email)

        model_input = tokenizer(temp_gpt_email, return_tensors="pt", padding=True, truncation=True)
        max_length = int((model_input['input_ids'].shape[1])/2)
        truncate_model_input = tokenizer(temp_gpt_email, return_tensors="pt", padding=True, truncation=True, max_length = max_length)
        truncate_model_input_string = tokenizer.batch_decode(truncate_model_input['input_ids'], skip_special_tokens=True)[0]

        truncate_gpt_email_list.append(truncate_model_input_string)
        
        input_string = system_prompt + " " + truncate_model_input_string

        length_input_string = len(input_string)

        truncate_model_input = tokenizer(input_string, return_tensors="pt").to(device='cuda')

        # for llama2
        # del truncate_model_input["token_type_ids"]

        model.eval()

        generate_text_length = 100
        with torch.no_grad():
            template_output_ids = model.generate(**truncate_model_input,
                                                max_new_tokens=generate_text_length,
                                                num_beams=3,
                                                pad_token_id=tokenizer.pad_token_id,
                                                num_return_sequences=1,
                                                early_stopping=True,
                                                eos_token_id=tokenizer.pad_token_id)

            template_output_batch = tokenizer.batch_decode(template_output_ids, skip_special_tokens=True)
            template_output_batch = template_output_batch[0]
            remain_string = template_output_batch[length_input_string:]

            remain_gpt_email_list.append(remain_string)
            
    df = pd.DataFrame({
        'original_gpt_email': original_gpt_email_list,
        'truncate_gpt_email': truncate_gpt_email_list,
        'remain_gpt_email': remain_gpt_email_list
    }, columns=['original_gpt_email', 'truncate_gpt_email', 'remain_gpt_email'])

    df_data = datasets.Dataset.from_pandas(df)
    df_data.save_to_disk(output_filepath)
    print("save done df_data")
    print(df_data)
    print("save done output_filepath:", output_filepath)