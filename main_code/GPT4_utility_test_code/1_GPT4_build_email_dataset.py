import torch
import os
import pandas as pd
import datasets
import argparse
from pathlib import Path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_filedir", type=str, default="./data/source", help="input_filedir")
    parser.add_argument("--output_filepath", type=str, default="./data/source/gpt_email_dataset", help="output directory")

    args = parser.parse_args()

    input_filedir = args.input_filedir
    output_filepath = args.output_filepath

    input_file_path_list = ["general_emails_10_final_part1.txt", "general_emails_10_final_part2.txt", "general_emails_10_final_part3.txt", "general_emails_10_final_part4.txt"]

    all_gpt4_generated_email_list = []
    for index in range(len(input_file_path_list)):
        file_path = input_file_path_list[index]
        full_file_path = os.path.join(input_filedir, file_path)
        gpt4_generated_email = Path(full_file_path).read_text().strip()
        gpt4_generated_email_list = gpt4_generated_email.split("---")
        gpt4_generated_email_list = gpt4_generated_email_list[:-1]
        all_gpt4_generated_email_list += gpt4_generated_email_list

    df = pd.DataFrame({
        'gpt_email': all_gpt4_generated_email_list
    }, columns=['gpt_email'])

    df_data = datasets.Dataset.from_pandas(df)

    df_data.save_to_disk(output_filepath)