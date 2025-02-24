import torch
from transformers import (
    set_seed,
)
import transformers
import torch
import numpy as np
import os
import pandas as pd
import datasets
from transformers import T5ForConditionalGeneration, T5Tokenizer
from tqdm import tqdm
import time
import argparse
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize

transformers.logging.set_verbosity_error()

set_seed(0)  # for reproducibility
# Set the random seed for NumPy
np.random.seed(0)
# Set the random seed for PyTorch
torch.manual_seed(0)
# If you are using CUDA (i.e., a GPU), also set the seed for it
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DipperParaphraser(object):
    def __init__(self, model="kalpeshk2011/dipper-paraphraser-xxl", verbose=True):
        time1 = time.time()
        self.tokenizer = T5Tokenizer.from_pretrained('google/t5-v1_1-xxl')
        self.model = T5ForConditionalGeneration.from_pretrained(model, device_map="auto")
        # self.model = T5ForConditionalGeneration.from_pretrained(model)
        # self.model = self.model.to(device)
        # self.model = self.model.cuda()
        if verbose:
            print(f"{model} model loaded in {time.time() - time1}")
        self.model.eval()

    # def paraphrase(self, input_text, lex_diversity, order_diversity, prefix="", sent_interval=3, **kwargs):
    def paraphrase(self, input_text, lex_diversity, order_diversity, sent_interval=3, **kwargs):
        """Paraphrase a text using the DIPPER model.

        Args:
            input_text (str): The text to paraphrase. Make sure to mark the sentence to be paraphrased between <sent> and </sent> blocks, keeping space on either side.
            lex_diversity (int): The lexical diversity of the output, choose multiples of 20 from 0 to 100. 0 means no diversity, 100 means maximum diversity.
            order_diversity (int): The order diversity of the output, choose multiples of 20 from 0 to 100. 0 means no diversity, 100 means maximum diversity.
            **kwargs: Additional keyword arguments like top_p, top_k, max_length.
        """
        assert lex_diversity in [0, 20, 40, 60, 80, 100], "Lexical diversity must be one of 0, 20, 40, 60, 80, 100."
        assert order_diversity in [0, 20, 40, 60, 80, 100], "Order diversity must be one of 0, 20, 40, 60, 80, 100."

        lex_code = int(100 - lex_diversity)
        order_code = int(100 - order_diversity)

        input_text = " ".join(input_text.split())
        sentences = sent_tokenize(input_text)
        output_text = ""

        for sent_idx in range(0, len(sentences), sent_interval):
            curr_sent_window = " ".join(sentences[sent_idx:sent_idx + sent_interval])
            final_input_text = f"lexical = {lex_code}, order = {order_code}"
            final_input_text += f" <sent> {curr_sent_window} </sent>"

            final_input = self.tokenizer([final_input_text], return_tensors="pt")
            # final_input = {k: v.cuda() for k, v in final_input.items()}
            # final_input = {k: v.to(device) for k, v in final_input.items()}
            final_input = {k: v.to('cuda:0') if torch.cuda.is_available() else v for k, v in final_input.items()}

            with torch.inference_mode():
                outputs = self.model.generate(**final_input, **kwargs)
            outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            output_text += " " + outputs[0]

        return output_text


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--attack_dataset_path", type=str, default="", help="attack_dataset_path")
    parser.add_argument("--tokenizer_name", type=str, default="meta-llama/Llama-2-7b-hf", help="tokenizer_name")
    parser.add_argument("--output_dir", type=str, default="./data/", help="output_dir")
    parser.add_argument("--output_filename", type=str, default="./data/", help="output_filename")
    args = parser.parse_args()

    attack_dataset_path = args.attack_dataset_path
    print("attack_dataset_path:", attack_dataset_path)
    tokenizer_name = args.tokenizer_name
    output_dir = args.output_dir
    print("output_dir:", output_dir)
    output_filename = args.output_filename
    print("output_filename:", output_filename)


    train_users_twin_phone_template_dataset = datasets.load_from_disk(attack_dataset_path)

    whole_template_dataset = train_users_twin_phone_template_dataset

    print("whole_template_dataset")
    print(whole_template_dataset)

    global_name_twin_phone_list = []
    global_category_twin_phone_list = []
    global_twin_phone_template_list = []
    global_twin_phone_template_type_list = []
    global_twin_phone_template_target_list = []
    global_twin_phone_target_pii_type_list = []

    dp = DipperParaphraser()

    for index in tqdm(range(len(whole_template_dataset))):
        name = whole_template_dataset[index]['NAME']
        category = whole_template_dataset[index]['CATEGORY']
        template = whole_template_dataset[index]['TEMPLATE']
        template_type = whole_template_dataset[index]['TEMPLATE_TYPE']
        target = whole_template_dataset[index]['TARGET']
        target_pii_type = whole_template_dataset[index]['TARGET_PII_TYPE']

        parap_list = []
        for t_temp in template:
            count = 0
            for lex_diversity in [20, 40, 60, 80]:
                for order_diversity in [20, 40, 60, 80]:
                    for top_p in [0.25, 0.5, 0.75]:
                        parap = dp.paraphrase(t_temp, lex_diversity=lex_diversity, order_diversity=order_diversity, do_sample=True, top_p=top_p, top_k=None, max_length=512)
                        parap_list.append(parap)
                        count += 1
                        if (count >= 4):
                            break
                    if (count >= 4):
                        break
                if (count >= 4):
                    break

        global_name_twin_phone_list.append(name)
        global_category_twin_phone_list.append(category)
        global_twin_phone_template_list.append(parap_list)
        global_twin_phone_template_type_list.append(template_type)
        global_twin_phone_template_target_list.append(target)
        global_twin_phone_target_pii_type_list.append(target_pii_type)

    df = pd.DataFrame({
        'NAME': global_name_twin_phone_list,
        'CATEGORY': global_category_twin_phone_list,
        'TEMPLATE': global_twin_phone_template_list,
        'TEMPLATE_TYPE': global_twin_phone_template_type_list,
        'TARGET': global_twin_phone_template_target_list,
        'TARGET_PII_TYPE': global_twin_phone_target_pii_type_list
    }, columns=['NAME', 'CATEGORY', 'TEMPLATE', 'TEMPLATE_TYPE', 'TARGET', 'TARGET_PII_TYPE'])

    df_data = datasets.Dataset.from_pandas(df)

    data_path = os.path.join(output_dir, output_filename)

    df_data.save_to_disk(data_path)