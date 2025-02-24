import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
    AdamW,
    get_cosine_schedule_with_warmup,
)
import boto3
import transformers
# from transformers import , AutoTokenizer, pipeline
import torch
import numpy as np
import os
import datasets
from typing import Any, Dict, List
import contextlib
from transformers import DataCollatorForLanguageModeling, Trainer, Seq2SeqTrainingArguments
import tqdm
import argparse
from torch.nn import CrossEntropyLoss

transformers.logging.set_verbosity_error()


set_seed(0)  # for reproducibility
# Set the random seed for NumPy
np.random.seed(0)
# Set the random seed for PyTorch
torch.manual_seed(0)
# If you are using CUDA (i.e., a GPU), also set the seed for it
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True

# select dataset index range
# selected_range = 120
# selected_range = 8

# aws service load
client = boto3.client('comprehend')

import functools

@contextlib.contextmanager
def main_process_first():
    """
    A context manager for torch distributed environment where on needs to do something on the main process, while
    blocking replicas, and when it's finished releasing the replicas.
    One such use is for `datasets`'s `map` feature which to be efficient should be run once on the main process,
    which upon completion saves a cached version of results and which then automatically gets loaded by the
    replicas.

    This is a stripped-down version of the the huggingface context manager from commit 2eb7bb15e771f13192968cd4657c78f76b0799fe
    """
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


# max_length = 128
# unlearning dataset avg length 19.35
# max_length = 20
# unlearning dataset max length 33

# martinc specific target name
# specific_target_name = "Steve Douglas"
# specific_target_name = "Teresa A. Callahan"

def preprocess_dataset(
    dataset,
    tokenizer
):
    


    def preprocess_pretrain_dataset(examples: Dict[str, List[Any]]) -> Dict[str, Any]:
        # build grouped texts with format `X1 X2 X3 ...`
        """
        kwargs = dict(add_special_tokens=True)

        if hasattr(tokenizer, "add_eos_token"): # for LLaMA tokenizer
            setattr(tokenizer, "add_eos_token", True)

        tokenized_examples = tokenizer(examples["TEXT"], **kwargs)
        """
        """
        kwargsaaa = dict(add_special_tokens=True)

        if hasattr(tokenizer, "add_eos_token"): # for LLaMA tokenizer
            setattr(tokenizer, "add_eos_token", True)
        """


        # martinc new
        max_length = 34

        print('examples["FAKE_PII_TEMPLATE"][0]')
        print(examples["FAKE_PII_TEMPLATE"][0])
        print('examples["FAKE_PII_TARGET"][0]')
        print(examples["FAKE_PII_TARGET"][0])

        fake_template_tokenized_examples = tokenizer(examples["FAKE_PII_TEMPLATE"], text_target=examples["FAKE_PII_TEMPLATE"], max_length=max_length, truncation=True)
        # FAKE_PII_TARGET
        fake_target_tokenized_examples = tokenizer(examples["FAKE_PII_TARGET"], text_target=examples["FAKE_PII_TARGET"], max_length=max_length, truncation=True, add_special_tokens=False)

        # print("tokenized_examples.keys()")
        # print(tokenized_examples.keys())# dict_keys(['input_ids', 'attention_mask', 'labels'])

        # print("fake_tokenized_examples.keys()")
        # print(fake_tokenized_examples.keys()) 


        # handle fake pii sub dataset tokenized_examples-----------------------------------------------------------------------
        
        """
        # when llamatokenizer tokenize labels, it may add space in the first position of the label, so we need to remove it, 
        # if there is a "space" at the begining of the labels, we will remove it.
        space_id = tokenizer.convert_tokens_to_ids('▁')

        for index in range(len(fake_target_tokenized_examples["labels"])):
            temp_target_input_ids = fake_target_tokenized_examples["input_ids"][index]
            temp_target_attention_mask = fake_target_tokenized_examples["attention_mask"][index]
            temp_target_labels = fake_target_tokenized_examples["labels"][index]
            # print("temp_target_labels-----------------")
            # print(temp_target_labels)
            if (temp_target_input_ids[0] == space_id):
                temp_target_input_ids = temp_target_input_ids[1:]
                temp_target_attention_mask = temp_target_attention_mask[1:]
                temp_target_labels = temp_target_labels[1:]
                # print("truncate temp_target_labels")
                # print(temp_target_labels)
                fake_target_tokenized_examples["input_ids"][index] = temp_target_input_ids
                fake_target_tokenized_examples["attention_mask"][index] = temp_target_attention_mask
                fake_target_tokenized_examples["labels"][index] = temp_target_labels
        """

        # handle fake pii sub dataset tokenized_examples-----------------------------------------------------------------------
        ignore_index = -100
        # hope_designed_forget_index = 2 # index for numbers position 
        # isnumeric = labels_to_tokens[l_index].isnumeric()
        for index in range(len(fake_target_tokenized_examples["labels"])):
            temp_target_labels = fake_target_tokenized_examples["labels"][index]
            labels_to_tokens = tokenizer.convert_ids_to_tokens(temp_target_labels)
            modified_temp_target_labels = [ignore_index] * len(temp_target_labels)

            for target_index in range(len(labels_to_tokens)):
                # if(labels_to_tokens[target_index].isnumeric()):
                modified_temp_target_labels[target_index] = temp_target_labels[target_index]
                

            fake_target_tokenized_examples["labels"][index] = modified_temp_target_labels

        for index in range(len(fake_template_tokenized_examples["labels"])):
            temp_labels = fake_template_tokenized_examples["labels"][index]
            modified_temp_labels = [ignore_index] * len(temp_labels)
            fake_template_tokenized_examples["labels"][index] = modified_temp_labels

        # concatenate unlearning tokenized_examples  
        for index in range(len(fake_template_tokenized_examples["input_ids"])):
            temp_template_input_ids = fake_template_tokenized_examples["input_ids"][index]
            temp_template_attention_mask = fake_template_tokenized_examples["attention_mask"][index]
            temp_template_labels = fake_template_tokenized_examples["labels"][index]

            temp_target_input_ids = fake_target_tokenized_examples["input_ids"][index]
            temp_target_attention_mask = fake_target_tokenized_examples["attention_mask"][index]
            temp_target_labels = fake_target_tokenized_examples["labels"][index]

            concatenate_input_ids = temp_template_input_ids + temp_target_input_ids
            concatenate_attention_mask = temp_template_attention_mask + temp_target_attention_mask
            concatenate_labels = temp_template_labels + temp_target_labels
    
            fake_template_tokenized_examples["input_ids"][index] = concatenate_input_ids
            fake_template_tokenized_examples["attention_mask"][index] = concatenate_attention_mask
            fake_template_tokenized_examples["labels"][index] = concatenate_labels
            
            # martinc test print
            invert_index_concatenate_input_ids = tokenizer.convert_ids_to_tokens(concatenate_input_ids)
            invert_index_temp_template_input_ids = tokenizer.convert_ids_to_tokens(temp_template_input_ids)
            invert_index_temp_target_input_ids = tokenizer.convert_ids_to_tokens(temp_target_input_ids)
            
            print("invert_index_concatenate_input_ids----------------------------")
            print(invert_index_concatenate_input_ids)
            print("invert_index_temp_template_input_ids")
            print(invert_index_temp_template_input_ids)
            print("invert_index_temp_target_input_ids")
            print(invert_index_temp_target_input_ids)
            print("concatenate_input_ids")
            print(concatenate_input_ids)
            print("temp_template_input_ids")
            print(temp_template_input_ids)
            print("temp_target_input_ids")
            print(temp_target_input_ids)
            print("concatenate_attention_mask")
            print(concatenate_attention_mask)
            print("concatenate_labels")
            print(concatenate_labels)
            larger_zero_index_list = [i for i in concatenate_labels if i > 0]
            larger_zero_labels = tokenizer.convert_ids_to_tokens(larger_zero_index_list)
            print("larger_zero_labels")
            print(larger_zero_labels)
            print("len(invert_index_concatenate_input_ids):", len(invert_index_concatenate_input_ids))
            print("len(concatenate_attention_mask):", len(concatenate_attention_mask))
            print("len(concatenate_labels):", len(concatenate_labels))
            

        fake_template_tokenized_examples['fake_input_ids'] = fake_template_tokenized_examples['input_ids']
        del fake_template_tokenized_examples['input_ids']

        fake_template_tokenized_examples['fake_attention_mask'] = fake_template_tokenized_examples['attention_mask']
        del fake_template_tokenized_examples['attention_mask']

        fake_template_tokenized_examples['fake_labels'] = fake_template_tokenized_examples['labels']
        del fake_template_tokenized_examples['labels']

        # template_tokenized_examples.update(fake_template_tokenized_examples)
        """
        #### martinc test check print
        for index in range(len(template_tokenized_examples['input_ids'])):

            temp_input_ids = template_tokenized_examples['input_ids'][index]
            temp_fake_input_ids = template_tokenized_examples['fake_input_ids'][index]

            temp_input_tokens = tokenizer.convert_ids_to_tokens(template_tokenized_examples['input_ids'][index])
            temp_fake_input_tokens = tokenizer.convert_ids_to_tokens(template_tokenized_examples['fake_input_ids'][index])

            temp_labels = template_tokenized_examples['labels'][index]
            temp_fake_labels = template_tokenized_examples['fake_labels'][index]

            print("temp_input_tokens-------------------------------")
            print(temp_input_tokens)
            print("temp_input_ids")
            print(temp_input_ids)
            print("temp_labels")
            print(temp_labels)
            print("temp_fake_input_tokens")
            print(temp_fake_input_tokens)
            print("temp_fake_input_ids")
            print(temp_fake_input_ids)
            print("temp_fake_labels")
            print(temp_fake_labels)
        """

        # return template_tokenized_examples
        return fake_template_tokenized_examples
    

    print("original dataset")
    print(dataset)
    # dataset = dataset.filter(lambda example: example["TEXT"])
    """
    Dataset({
        features: ['NAME', 'CATEGORY', 'TEMPLATE', 'TEMPLATE_TYPE', 'TARGET', 'TARGET_PII_TYPE'],
        num_rows: 577
    })
    """

    preprocess_func = preprocess_pretrain_dataset

    # with training_args.main_process_first(desc="dataset map pre-processing"):
    with main_process_first():
        
        kwargs = {}

        num_threads = get_num_workers()

        # if not data_args.streaming:
        kwargs = dict(
            # num_proc=data_args.preprocessing_num_workers,
            num_proc=num_threads if num_threads > 0 else None,
            load_from_cache_file=False,
            # load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset"
        )


        dataset = dataset.rename_column("TEMPLATE", "FAKE_PII_TEMPLATE")
        dataset = dataset.rename_column("TARGET", "FAKE_PII_TARGET")

        print("modified dataset")
        print(dataset)

        """
        new_column = ["n_empty"] * len(dataset)
        dataset = dataset.add_column("FAKE_PII_TEMPLATE", new_column)

        new_column = ["n_empty"] * len(dataset)
        dataset = dataset.add_column("FAKE_PII_TARGET", new_column)
        """
        column_names = list(next(iter(dataset)).keys())

        print("before preprocess_func")
        dataset = dataset.map(
            preprocess_func,
            batched=True,            
            remove_columns=column_names,
            **kwargs
        )
        print("after preprocess_func")
        dataset_list = list(iter(dataset))

        print("len(dataset_list):", len(dataset_list))
        
    
    return dataset

def collate(batch, pad_index):
    # dict_keys(['input_ids', 'attention_mask', 'labels', 'fake_input_ids', 'fake_attention_mask', 'fake_labels'])

    fake_input_ids = [torch.LongTensor(i['fake_input_ids']) for i in batch]
    # batch_ids = torch.nn.utils.rnn.pad_sequence(batch_ids, padding_value=pad_index, batch_first=True)
    fake_input_ids = torch.nn.utils.rnn.pad_sequence(fake_input_ids, padding_value=pad_index, batch_first=True)
    
    fake_attention_mask = [torch.LongTensor(i['fake_attention_mask']) for i in batch]
    # attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, padding_value=0, batch_first=True)
    fake_attention_mask = torch.nn.utils.rnn.pad_sequence(fake_attention_mask, padding_value=0, batch_first=True)
    
    fake_labels = [torch.LongTensor(i['fake_labels']) for i in batch]
    fake_labels = torch.nn.utils.rnn.pad_sequence(fake_labels, padding_value=-100, batch_first=True)

    # fake_labels = torch.LongTensor([i['fake_labels'] for i in batch])

    # batch = {'ids': batch_ids, 'length': batch_length, 'label': batch_label}
    batch = {'fake_input_ids': fake_input_ids, 'fake_attention_mask': fake_attention_mask, 'fake_labels': fake_labels}
    return batch

collate_fn = collate

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="MartinKu/", help="model_name")# MartinKu/bookcorpus_SV
parser.add_argument("--tokenizer_name", type=str, default="meta-llama/Llama-2-7b-hf", help="tokenizer_name")
parser.add_argument("--fake_data_path", type=str, default="MartinKu/", help="fake_data_path")# MartinKu/bookcorpus_SV
# num_train_epochs
parser.add_argument("--num_train_epochs", type=int, default=0)
parser.add_argument("--output_filepath", type=str, default="./data/", help="output directory")
# filepath = "./propile_blackbox_attack_results/only_pii_collapse_defense_steve_douglas/2024127_only_defense_steve_douglas_attack_rank_" + str(rank) + "_.csv"

args = parser.parse_args()
# martinc pii collapse
# model_name = "./specific_users_collapse_pii_defense/results_final_2024122_Steve_Douglas_llama2_7b_whole_enron_defense_e1_wd_0.001_lr_5e-05_b_1"
model_name = args.model_name
# model_name = "./results_llama2_7b_whole_enron/checkpoint-269696"
print("model_name:", model_name)
tokenizer_name = args.tokenizer_name
fake_data_path = args.fake_data_path
print("fake_data_path:", fake_data_path)

output_filepath = args.output_filepath
print("output_filepath:", output_filepath)

num_train_epochs = args.num_train_epochs


# Load LLaMA tokenizer
# tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True, use_fast=False, clean_up_tokenization_spaces=True, add_prefix_space=False)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)# clean_up_tokenization_spaces decode_with_prefix_space=False
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training

print("LLaMA tokenizer-----------------------------------------------------------------")
print(tokenizer)

fake_dataset = datasets.load_from_disk(fake_data_path)

print("fake_dataset")
print(fake_dataset)

# dataset preprocess_dataset
fake_dataset = preprocess_dataset(fake_dataset, tokenizer)
print("preprocess_dataset done")



# Load base model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    # quantization_config=bnb_config,
    # device_map=device_map
    device_map="auto"
)
model.config.use_cache = False
model.config.pretraining_tp = 1

print("model------------------------------------")
print(model)



data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


# Output directory where the model predictions and checkpoints will be stored
# output_dir = "./softprompt_adv_train_dir"
# output_dir = "../selected_unlearning_softprompt_adv_train_dir/selected_unlearning_softprompt_adv_train_epoch_0"
 
 
# Enable fp16/bf16 training (set bf16 to True with an A100)
fp16 = False
bf16 = False
 
# Batch size per GPU for training
per_device_train_batch_size = 4 # 16 # 4
 
# Batch size per GPU for evaluation
per_device_eval_batch_size = 4 # 16 # 4
 
# Number of update steps to accumulate the gradients for
gradient_accumulation_steps = 1
 
# Enable gradient checkpointing
gradient_checkpointing = True
 
# Maximum gradient normal (gradient clipping)
# martin original open
max_grad_norm = 0.3
 
# Initial learning rate (AdamW optimizer)
learning_rate = 5e-5 # 2e-4 5e-5
 
# Weight decay to apply to all layers except bias/LayerNorm weights
weight_decay = 0.001
 
# Optimizer to use
optim = "paged_adamw_32bit"
 
# Learning rate schedule
lr_scheduler_type = "cosine"
 
# Number of training steps (overrides num_train_epochs)
max_steps = -1
 
# Ratio of steps for a linear warmup (from 0 to learning rate)
warmup_ratio = 0.03
# warmup_ratio = 0
 
# Group sequences into batches with same length
# Saves memory and speeds up training considerably
group_by_length = True
 
# Save checkpoint every X updates steps
# save_steps = 0

# len_dataset_list = len(list(dataset))
# save_steps = (len_dataset_list // per_device_train_batch_size) + 1
# print("martinc save_steps:", save_steps, " per_device_train_batch_size:", per_device_train_batch_size, " len_dataset_list:", len_dataset_list)


# Log every X updates steps
logging_steps = 1


### traditional 
collate = functools.partial(collate_fn, pad_index=tokenizer.pad_token_id)
dataloader = torch.utils.data.DataLoader(
    dataset=fake_dataset,
    batch_size=per_device_train_batch_size,
    collate_fn=collate
    )


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model.train()
epoch_losses = []
epoch_accs = []

criterion = CrossEntropyLoss()
criterion = criterion.to(device)

optimizer = AdamW(
    model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(dataloader))  # PyTorch scheduler
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(dataloader) * num_train_epochs)  # PyTorch scheduler

for epoch in range(num_train_epochs):
    # for batch in tqdm.tqdm(dataloader, desc='training...', file=sys.stdout):
    for batch in tqdm.tqdm(dataloader, desc='training...'):
        lr_rate = scheduler.get_last_lr()

        fake_input_ids = batch['fake_input_ids'].to(device)
        fake_attention_mask = batch['fake_attention_mask'].to(device)
        fake_labels = batch['fake_labels'].to(device)

        outputs = model(fake_input_ids, attention_mask=fake_attention_mask, labels=fake_labels)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()

        print("lr_rate:", lr_rate, " after descent backward loss:", loss)

# model.save_pretrained("/home/my_name/Desktop/t5small", from_pt=True) 

print("before save model")
# ../selected_unlearning_max_and_fake_pii_min_dir/test/
# save_model_path = "../selected_unlearning_max_and_fake_pii_min_dir/test/min_max_test"
# save_model_path = "../selected_unlearning_max_and_fake_pii_min_dir/softprompt_adv_train_dir/adv_epoch_0/e_1_defense_fake_pii"
model.save_pretrained(output_filepath, from_pt=True)
print("done save model output_filepath:", output_filepath)

print("all quit---------------------------------")
quit()