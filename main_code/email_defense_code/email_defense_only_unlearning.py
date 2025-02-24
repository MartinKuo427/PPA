import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
    AutoConfig,
    AdamW,
    get_cosine_schedule_with_warmup,
)
import boto3
import transformers
import torch
import numpy as np
import os
import pandas as pd
import datasets
from typing import Any, Dict, List
import contextlib
from transformers import DataCollatorForLanguageModeling, Trainer, Seq2SeqTrainingArguments
import tqdm
import time
import argparse
from torch.nn import CrossEntropyLoss
import copy

transformers.logging.set_verbosity_error()


set_seed(0)  # for reproducibility
# Set the random seed for NumPy
np.random.seed(0)
# Set the random seed for PyTorch
torch.manual_seed(0)
# If you are using CUDA (i.e., a GPU), also set the seed for it
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True

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

# combine_dataset = preprocess_dataset(unlearning_dataset, fake_dataset, tokenizer)

def calculate_perplexity_preprocess_dataset(
    dataset,
    tokenizer
):

    column_names = list(next(iter(dataset)).keys())


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

        template_tokenized_examples = tokenizer(examples["TEMPLATE"], text_target=examples["TEMPLATE"], max_length=max_length, truncation=True)
        target_tokenized_examples = tokenizer(examples["TARGET"], text_target=examples["TARGET"], max_length=max_length, truncation=True, add_special_tokens=False)

        # print("tokenized_examples.keys()")
        # print(tokenized_examples.keys())# dict_keys(['input_ids', 'attention_mask', 'labels'])

        # print("fake_tokenized_examples.keys()")
        # print(fake_tokenized_examples.keys()) 


        # handle unlearning dataset tokenized_examples-----------------------------------------------------------------------
        
        
        # when llamatokenizer tokenize labels, it may add space in the first position of the label, so we need to remove it, 
        # if there is a "space" at the begining of the labels, we will remove it.
        space_id = tokenizer.convert_tokens_to_ids('▁')

        for index in range(len(target_tokenized_examples["labels"])):
            temp_target_input_ids = target_tokenized_examples["input_ids"][index]
            temp_target_attention_mask = target_tokenized_examples["attention_mask"][index]
            temp_target_labels = target_tokenized_examples["labels"][index]
            # print("temp_target_labels-----------------")
            # print(temp_target_labels)
            if (temp_target_input_ids[0] == space_id):
                temp_target_input_ids = temp_target_input_ids[1:]
                temp_target_attention_mask = temp_target_attention_mask[1:]
                temp_target_labels = temp_target_labels[1:]
                # print("truncate temp_target_labels")
                # print(temp_target_labels)
                target_tokenized_examples["input_ids"][index] = temp_target_input_ids
                target_tokenized_examples["attention_mask"][index] = temp_target_attention_mask
                target_tokenized_examples["labels"][index] = temp_target_labels
        
        
        template_tokenized_examples["remain_attention_mask"] = template_tokenized_examples["attention_mask"]
        template_tokenized_examples["begin_pii_index"] = template_tokenized_examples["attention_mask"]
        
        ignore_index = -100

        for index in range(len(template_tokenized_examples["labels"])):
            temp_labels = template_tokenized_examples["labels"][index]
            modified_temp_labels = [ignore_index] * len(temp_labels)
            template_tokenized_examples["labels"][index] = modified_temp_labels

        # concatenate unlearning tokenized_examples  
        for index in range(len(template_tokenized_examples["input_ids"])):
            temp_template_input_ids = template_tokenized_examples["input_ids"][index]
            temp_template_attention_mask = template_tokenized_examples["attention_mask"][index]
            temp_template_labels = template_tokenized_examples["labels"][index]

            temp_target_input_ids = target_tokenized_examples["input_ids"][index]
            temp_target_attention_mask = target_tokenized_examples["attention_mask"][index]
            temp_target_labels = target_tokenized_examples["labels"][index]

            concatenate_input_ids = temp_template_input_ids + temp_target_input_ids
            concatenate_attention_mask = temp_template_attention_mask + temp_target_attention_mask
            concatenate_labels = temp_template_labels + temp_target_labels

            remain_concatenate_attention_mask = np.copy(concatenate_attention_mask)
            # remain_concatenate_attention_mask = concatenate_attention_mask.copy()
            remain_concatenate_attention_mask[:len(temp_template_attention_mask)] = 0
            remain_concatenate_attention_mask = list(remain_concatenate_attention_mask)

            template_tokenized_examples["input_ids"][index] = concatenate_input_ids
            template_tokenized_examples["attention_mask"][index] = concatenate_attention_mask
            template_tokenized_examples["labels"][index] = concatenate_labels
            template_tokenized_examples["remain_attention_mask"][index] = remain_concatenate_attention_mask
            template_tokenized_examples["begin_pii_index"][index] = len(temp_template_input_ids)
            
            """
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
            print("remain_concatenate_attention_mask")
            print(remain_concatenate_attention_mask)
            print("concatenate_labels")
            print(concatenate_labels)
            print("len(temp_template_input_ids):", len(temp_template_input_ids))
            larger_zero_index_list = [i for i in concatenate_labels if i > 0]
            larger_zero_labels = tokenizer.convert_ids_to_tokens(larger_zero_index_list)
            print("larger_zero_labels")
            print(larger_zero_labels)
            print("len(invert_index_concatenate_input_ids):", len(invert_index_concatenate_input_ids))
            print("len(concatenate_attention_mask):", len(concatenate_attention_mask))
            print("len(concatenate_labels):", len(concatenate_labels))
            """

        return template_tokenized_examples
    

    print("original dataset")
    print(dataset)
    # dataset = dataset.filter(lambda example: example["TEXT"])
    print("after filter dataset")
    print(dataset)

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


# preprocess_dataset(ori_combine_dataset, tokenizer)
def preprocess_dataset(
    ori_combine_dataset,
    tokenizer
):
    # features: ['NAME', 'TEMPLATE', 'TARGET', 'SELECT_UNLEARNING_INDEX', 'FAKE_TEMPLATE', 'FAKE_TARGET', 'GRADIENT_COMPENSATE_FLAG']
    
    def preprocess_pretrain_dataset(examples: Dict[str, List[Any]]) -> Dict[str, Any]:
        # build grouped texts with format `X1 X2 X3 ...`

        # select_unlearning_index_list = examples["SELECT_UNLEARNING_INDEX"]

        # martinc new
        max_length = 34

        template_tokenized_examples = tokenizer(examples["TEMPLATE"], text_target=examples["TEMPLATE"], max_length=max_length, truncation=True)
        target_tokenized_examples = tokenizer(examples["TARGET"], text_target=examples["TARGET"], max_length=max_length, truncation=True, add_special_tokens=False)
    
        # print("tokenized_examples.keys()")
        # print(tokenized_examples.keys())# dict_keys(['input_ids', 'attention_mask', 'labels'])

        # print("fake_tokenized_examples.keys()")
        # print(fake_tokenized_examples.keys()) 


        # handle unlearning dataset tokenized_examples-----------------------------------------------------------------------
        
        # when llamatokenizer tokenize labels, it may add space in the first position of the label, so we need to remove it, 
        # if there is a "space" at the begining of the labels, we will remove it.
        space_id = tokenizer.convert_tokens_to_ids('▁')

        for index in range(len(target_tokenized_examples["labels"])):
            temp_target_input_ids = target_tokenized_examples["input_ids"][index]
            temp_target_attention_mask = target_tokenized_examples["attention_mask"][index]
            temp_target_labels = target_tokenized_examples["labels"][index]
            # print("temp_target_labels-----------------")
            # print(temp_target_labels)
            if (temp_target_input_ids[0] == space_id):
                temp_target_input_ids = temp_target_input_ids[1:]
                temp_target_attention_mask = temp_target_attention_mask[1:]
                temp_target_labels = temp_target_labels[1:]
                # print("truncate temp_target_labels")
                # print(temp_target_labels)
                target_tokenized_examples["input_ids"][index] = temp_target_input_ids
                target_tokenized_examples["attention_mask"][index] = temp_target_attention_mask
                target_tokenized_examples["labels"][index] = temp_target_labels

            
        
        ignore_index = -100
        # hope_designed_forget_index = 1 # index for numbers position 
        # isnumeric = labels_to_tokens[l_index].isnumeric()

        # martinc
        # print_check_list = ["x33102", "3-9610", "x35275", "x54070", "713-957-7060", "713.345.4727", "5-7857", "3-3985", "x39510", "x36161", "416-352-4580", "3-3210", "39106", "58059"]

        # print_check_list = ["+583 330853915", "(713)853-9974", "(312) 682-2988", "x53240", "303-623-0987"]
        #                    

        print_check_list = ["+583 330853915", "(713)853-9974", "(312) 682-2988", "x53240)", "303-623-0987"]
        #                    [8],             [2],             [2],             [4]     , [6]

        for index in range(len(target_tokenized_examples["labels"])):

            #hope_designed_forget_index = select_unlearning_index_list[index]

            temp_target_labels = target_tokenized_examples["labels"][index]
            
            """
            # martinc test print
            temp_target_labels_string = tokenizer.decode(temp_target_labels)
            if (temp_target_labels_string in print_check_list):
            # if (index > (len(target_tokenized_examples["labels"]) - 6)):
                print("temp_target_labels_string-------------------:", temp_target_labels_string)
                print("hope_designed_forget_index:", hope_designed_forget_index)
            """
            labels_to_tokens = tokenizer.convert_ids_to_tokens(temp_target_labels)
            
            modified_temp_target_labels = [ignore_index] * len(temp_target_labels)

            """
            # foresee how many numeric in labels
            total_number_of_isnumeric = 0
            for target_index in range(len(labels_to_tokens)):
                if(labels_to_tokens[target_index].isnumeric()):
                    total_number_of_isnumeric += 1
            """
            # must unlearning one element in labels
            # designed_forget_index = min (hope_designed_forget_index, total_number_of_isnumeric)
            # designed_forget_index = hope_designed_forget_index

            for target_index in range(len(labels_to_tokens)):
                # if(labels_to_tokens[target_index].isnumeric()):
                # if (target_index in designed_forget_index):
                modified_temp_target_labels[target_index] = temp_target_labels[target_index]

            target_tokenized_examples["labels"][index] = modified_temp_target_labels

        for index in range(len(template_tokenized_examples["labels"])):
            temp_labels = template_tokenized_examples["labels"][index]
            modified_temp_labels = [ignore_index] * len(temp_labels)
            template_tokenized_examples["labels"][index] = modified_temp_labels

        # concatenate unlearning tokenized_examples  
        for index in range(len(template_tokenized_examples["input_ids"])):
            temp_template_input_ids = template_tokenized_examples["input_ids"][index]
            temp_template_attention_mask = template_tokenized_examples["attention_mask"][index]
            temp_template_labels = template_tokenized_examples["labels"][index]

            temp_target_input_ids = target_tokenized_examples["input_ids"][index]
            temp_target_attention_mask = target_tokenized_examples["attention_mask"][index]
            temp_target_labels = target_tokenized_examples["labels"][index]

            concatenate_input_ids = temp_template_input_ids + temp_target_input_ids
            concatenate_attention_mask = temp_template_attention_mask + temp_target_attention_mask
            concatenate_labels = temp_template_labels + temp_target_labels

            template_tokenized_examples["input_ids"][index] = concatenate_input_ids
            template_tokenized_examples["attention_mask"][index] = concatenate_attention_mask
            template_tokenized_examples["labels"][index] = concatenate_labels
            """
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
            """


        # martinc new
        max_length = 34

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
            
            """
            # martinc test print
            invert_index_concatenate_input_ids = tokenizer.convert_ids_to_tokens(concatenate_input_ids)
            invert_index_temp_template_input_ids = tokenizer.convert_ids_to_tokens(temp_template_input_ids)
            invert_index_temp_target_input_ids = tokenizer.convert_ids_to_tokens(temp_target_input_ids)
            
            print("fake invert_index_concatenate_input_ids----------------------------")
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
            """

        fake_template_tokenized_examples['fake_input_ids'] = fake_template_tokenized_examples['input_ids']
        del fake_template_tokenized_examples['input_ids']

        fake_template_tokenized_examples['fake_attention_mask'] = fake_template_tokenized_examples['attention_mask']
        del fake_template_tokenized_examples['attention_mask']

        fake_template_tokenized_examples['fake_labels'] = fake_template_tokenized_examples['labels']
        del fake_template_tokenized_examples['labels']

        template_tokenized_examples.update(fake_template_tokenized_examples)

        template_tokenized_examples['gradient_compensate_flag'] = examples["GRADIENT_COMPENSATE_FLAG"]

        # """
        #### martinc test check print
        for index in range(len(template_tokenized_examples['input_ids'])):

            temp_input_ids = template_tokenized_examples['input_ids'][index]
            temp_fake_input_ids = template_tokenized_examples['fake_input_ids'][index]

            temp_input_tokens = tokenizer.convert_ids_to_tokens(template_tokenized_examples['input_ids'][index])
            temp_fake_input_tokens = tokenizer.convert_ids_to_tokens(template_tokenized_examples['fake_input_ids'][index])

            temp_labels = template_tokenized_examples['labels'][index]
            temp_fake_labels = template_tokenized_examples['fake_labels'][index]

            temp_gradient_compensate_flag = template_tokenized_examples['gradient_compensate_flag'][index]

            print("combine temp_input_tokens-------------------------------")
            print(temp_input_tokens)
            print("temp_gradient_compensate_flag:", temp_gradient_compensate_flag)
            # print("temp_input_ids")
            # print(temp_input_ids)
            print("temp_labels")
            print(temp_labels)
            print("temp_fake_input_tokens")
            print(temp_fake_input_tokens)
            # print("temp_fake_input_ids")
            # print(temp_fake_input_ids)
            print("temp_fake_labels")
            print(temp_fake_labels)
        # """

        return template_tokenized_examples
        # return fake_template_tokenized_examples
    
    """
    print("unlearning_dataset")
    print(unlearning_dataset)
    print("fake_dataset")
    print(fake_dataset)
    """
    # dataset = dataset.filter(lambda example: example["TEXT"])
    """
    Dataset({
        features: ['NAME', 'CATEGORY', 'TEMPLATE', 'TEMPLATE_TYPE', 'TARGET', 'TARGET_PII_TYPE'],
        num_rows: 577
    })
    """

    preprocess_pretrain_func = preprocess_pretrain_dataset
    """
    fake_pii_preprocess_func = fake_pii_preprocess_pretrain_dataset
    unlearning_preprocess_func = unlearning_preprocess_pretrain_dataset
    """

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
        """
        fake_dataset = fake_dataset.rename_column("NAME", "FAKE_NAME")
        fake_dataset = fake_dataset.rename_column("TEMPLATE", "FAKE_PII_TEMPLATE")
        fake_dataset = fake_dataset.rename_column("TARGET", "FAKE_PII_TARGET")
        """
        # combine unlearning_dataset and fake_dataset to same row number
        # need to match with name
        # add new column to decide which row do gradient descent and which row do gradient ascent

        # combine fake_dataset into unlearning_dataset

        """
        print("unlearning_dataset")
        print(unlearning_dataset)
        print("modified fake_dataset")
        print(fake_dataset)
        # fake_dataset need to align with unlearning_dataset
        """

        """
        new_column = ["n_empty"] * len(dataset)
        dataset = dataset.add_column("FAKE_PII_TEMPLATE", new_column)

        new_column = ["n_empty"] * len(dataset)
        dataset = dataset.add_column("FAKE_PII_TARGET", new_column)
        """

        """
        column_names = list(next(iter(fake_dataset)).keys())

        print("before preprocess_func")
        fake_dataset = fake_dataset.map(
            fake_pii_preprocess_func,
            batched=True,            
            remove_columns=column_names,
            **kwargs
        )

        print("after fake_pii_preprocess_func fake_dataset")
        print(fake_dataset)


        column_names = list(next(iter(unlearning_dataset)).keys())

        print("before preprocess_func")
        unlearning_dataset = unlearning_dataset.map(
            unlearning_preprocess_func,
            batched=True,            
            remove_columns=column_names,
            **kwargs
        )
        """

        column_names = list(next(iter(ori_combine_dataset)).keys())

        combine_dataset = ori_combine_dataset.map(
            preprocess_pretrain_func,
            batched=True,            
            remove_columns=column_names,
            **kwargs
        )


        # combine unlearning_dataset and fake_dataset to same row number
        # need to match with name
        # add new column to decide which row do gradient descent and which row do gradient ascent

        # combine fake_dataset into unlearning_dataset
        # new_column = [0] * len(unlearning_dataset)
        # unlearning_dataset = unlearning_dataset.add_column("FAKE_PII_TARGET", new_column)

        """
        fake_template_tokenized_examples['fake_input_ids'] = fake_template_tokenized_examples['input_ids']
        del fake_template_tokenized_examples['input_ids']

        fake_template_tokenized_examples['fake_attention_mask'] = fake_template_tokenized_examples['attention_mask']
        del fake_template_tokenized_examples['attention_mask']

        fake_template_tokenized_examples['fake_labels'] = fake_template_tokenized_examples['labels']
        del fake_template_tokenized_examples['labels']
        """

        print("after preprocess_func")
        # dataset_list = list(iter(dataset))
        dataset_list = list(iter(combine_dataset))

        print("len(dataset_list):", len(dataset_list))
        
    
    # return dataset
    return combine_dataset

def collate(batch, pad_index):
    # dict_keys(['input_ids', 'attention_mask', 'labels', 'fake_input_ids', 'fake_attention_mask', 'fake_labels'])

    input_ids = [torch.LongTensor(i['input_ids']) for i in batch]
    # batch_ids = torch.nn.utils.rnn.pad_sequence(batch_ids, padding_value=pad_index, batch_first=True)
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, padding_value=pad_index, batch_first=True)
    
    attention_mask = [torch.LongTensor(i['attention_mask']) for i in batch]
    # attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, padding_value=0, batch_first=True)
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, padding_value=0, batch_first=True)
    
    labels = [torch.LongTensor(i['labels']) for i in batch]
    labels = torch.nn.utils.rnn.pad_sequence(labels, padding_value=-100, batch_first=True)


    fake_input_ids = [torch.LongTensor(i['fake_input_ids']) for i in batch]
    # batch_ids = torch.nn.utils.rnn.pad_sequence(batch_ids, padding_value=pad_index, batch_first=True)
    fake_input_ids = torch.nn.utils.rnn.pad_sequence(fake_input_ids, padding_value=pad_index, batch_first=True)
    
    fake_attention_mask = [torch.LongTensor(i['fake_attention_mask']) for i in batch]
    # attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, padding_value=0, batch_first=True)
    fake_attention_mask = torch.nn.utils.rnn.pad_sequence(fake_attention_mask, padding_value=0, batch_first=True)
    
    fake_labels = [torch.LongTensor(i['fake_labels']) for i in batch]
    fake_labels = torch.nn.utils.rnn.pad_sequence(fake_labels, padding_value=-100, batch_first=True)

    gradient_compensate_flag = torch.LongTensor([i['gradient_compensate_flag'] for i in batch])

    # batch = {'ids': batch_ids, 'length': batch_length, 'label': batch_label}
    batch = {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels, 
             'fake_input_ids': fake_input_ids, 'fake_attention_mask': fake_attention_mask, 'fake_labels': fake_labels,
             'gradient_compensate_flag': gradient_compensate_flag}
    return batch

collate_fn = collate

def calculate_perplexity_collate(batch, pad_index):
    # dict_keys(['input_ids', 'attention_mask', 'labels', 'fake_input_ids', 'fake_attention_mask', 'fake_labels'])

    input_ids = [torch.LongTensor(i['input_ids']) for i in batch]
    # batch_ids = torch.nn.utils.rnn.pad_sequence(batch_ids, padding_value=pad_index, batch_first=True)
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, padding_value=pad_index, batch_first=True)
    
    attention_mask = [torch.LongTensor(i['attention_mask']) for i in batch]
    # attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, padding_value=0, batch_first=True)
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, padding_value=0, batch_first=True)
    
    labels = [torch.LongTensor(i['labels']) for i in batch]
    labels = torch.nn.utils.rnn.pad_sequence(labels, padding_value=-100, batch_first=True)

    remain_attention_mask = [torch.LongTensor(i['remain_attention_mask']) for i in batch]
    # attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, padding_value=0, batch_first=True)
    remain_attention_mask = torch.nn.utils.rnn.pad_sequence(remain_attention_mask, padding_value=0, batch_first=True)

    begin_pii_index = torch.LongTensor([i['begin_pii_index'] for i in batch])


    # fake_labels = torch.LongTensor([i['fake_labels'] for i in batch])

    # batch = {'ids': batch_ids, 'length': batch_length, 'label': batch_label}
    batch = {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels, 'remain_attention_mask': remain_attention_mask, 'begin_pii_index': begin_pii_index}
    return batch

calculate_perplexity_collate_fn = calculate_perplexity_collate


parser = argparse.ArgumentParser()
# parser.add_argument("--input_dataset_path", type=str, default="MartinKu/", help="template_dataset path")# MartinKu/bookcorpus_SV
# parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument("--model_name", type=str, default="MartinKu/", help="model_name")# MartinKu/bookcorpus_SV
parser.add_argument("--tokenizer_name", type=str, default="meta-llama/Llama-2-7b-hf", help="tokenizer_name")
parser.add_argument("--unlearning_data_path", type=str, default="MartinKu/", help="unlearning_data_path")# MartinKu/bookcorpus_SV
parser.add_argument("--fake_data_path", type=str, default="MartinKu/", help="fake_data_path")# MartinKu/bookcorpus_SV
parser.add_argument("--ori_combine_dataset_path", type=str, default="MartinKu/", help="ori_combine_dataset_path")# MartinKu/bookcorpus_SV
# num_train_epochs
parser.add_argument("--num_train_epochs", type=int, default=0)
parser.add_argument("-f", "--forget_number_of_token", type=int, default=1)
# /home/mk585/2024_propile_7b_llama2_LLM_dataprivacy_ver1/huggingface_train_enron_email_dataset/propile_blackbox_template_dataset/allen_twin_phone_template_dataset
parser.add_argument("--output_filepath", type=str, default="./data/", help="output directory")
# filepath = "./propile_blackbox_attack_results/only_pii_collapse_defense_steve_douglas/2024127_only_defense_steve_douglas_attack_rank_" + str(rank) + "_.csv"

args = parser.parse_args()

# martinc pii collapse
# model_name = "./specific_users_collapse_pii_defense/results_final_2024122_Steve_Douglas_llama2_7b_whole_enron_defense_e1_wd_0.001_lr_5e-05_b_1"

model_name = args.model_name
# model_name = "./results_llama2_7b_whole_enron/checkpoint-269696"
print("model_name:", model_name)

tokenizer_name = args.tokenizer_name

forget_number_of_token = args.forget_number_of_token
print("forget_number_of_token:", forget_number_of_token)

unlearning_data_path = args.unlearning_data_path
print("unlearning_data_path:", unlearning_data_path)

fake_data_path = args.fake_data_path
print("fake_data_path:", fake_data_path)

ori_combine_dataset_path = args.ori_combine_dataset_path
print("ori_combine_dataset_path:", ori_combine_dataset_path)

output_filepath = args.output_filepath
print("output_filepath:", output_filepath)

num_train_epochs = args.num_train_epochs



PII_length_dict = {
    'address': 30,
    'phone number': 15
}

PII_length_key = list(PII_length_dict)

"""
print("whole_template_dataset---------------------")
print(whole_template_dataset)
"""

"""
Dataset({
    features: ['NAME', 'TEMPLATE', 'TARGET'],
    num_rows: 713
})
"""

# whole_template_dataset = whole_template_dataset.shuffle(seed=42)

"""
# for index in range(len(whole_template_dataset)):
for index in range(10):
    print("index--------------------:", index)
    print('whole_template_dataset["NAME"][index]:', whole_template_dataset["NAME"][index])
    print('whole_template_dataset["TEMPLATE"]:', whole_template_dataset["TEMPLATE"][index])
    print('whole_template_dataset["TARGET"]:', whole_template_dataset["TARGET"][index])
"""




# Load LLaMA tokenizer
# tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True, use_fast=False, clean_up_tokenization_spaces=True, add_prefix_space=False)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)# clean_up_tokenization_spaces decode_with_prefix_space=False
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training

print("LLaMA tokenizer-----------------------------------------------------------------")
print(tokenizer)


"""
LlamaTokenizerFast(name_or_path='NousResearch/Llama-2-7b-hf', vocab_size=32000, model_max_length=1000000000000000019884624838656, is_fast=True, padding_side='right', truncation_side='right', special_tokens={'bos_token': '<s>', 'eos_token': '</s>', 'unk_token': '<unk>', 'pad_token': '</s>'}, clean_up_tokenization_spaces=False),  added_tokens_decoder={
        0: AddedToken("<unk>", rstrip=False, lstrip=False, single_word=False, normalized=True, special=True),
        1: AddedToken("<s>", rstrip=False, lstrip=False, single_word=False, normalized=True, special=True),
        2: AddedToken("</s>", rstrip=False, lstrip=False, single_word=False, normalized=True, special=True),
}
"""



unlearning_dataset = datasets.load_from_disk(unlearning_data_path)

print("unlearning_dataset")
print(unlearning_dataset)

# ./propile_unlearning_template_dataset/20240229_unlearning_twin_phone_template_dataset

# unlearning_defense_train_data_path = "../propile_unlearning_template_dataset/20240229_unlearning_twin_phone_template_dataset"
fake_dataset = datasets.load_from_disk(fake_data_path)

print("fake_dataset")
print(fake_dataset)

"""
def replace_value(example):
    # if example['column_name'] == 'current_value':
    #     example['column_name'] = 'new_value'
    example['TEMPLATE'] = example['TEMPLATE'] + " "
    return example

fake_dataset = fake_dataset.map(replace_value)
"""


"""
fake_defense_train_data_path = "../propile_fake_template_dataset/20240227_fake_phone number_twin_phone_template_dataset"
print("fake_defense_train_data_path:", fake_defense_train_data_path)
fake_defense_train_dataset = datasets.load_from_disk(fake_defense_train_data_path)

print("fake_defense_train_dataset")
print(fake_defense_train_dataset)
"""


"""
fake_defense_train_dataset
Dataset({
    features: ['NAME', 'CATEGORY', 'TEMPLATE', 'TEMPLATE_TYPE', 'TARGET', 'TARGET_PII_TYPE'],
    num_rows: 206
})
"""
#### add space behind every template in unlearning_dataset
"""
print("unlearning_dataset['TEMPLATE'][0]")
print(unlearning_dataset['TEMPLATE'][0])
print("unlearning_dataset['TEMPLATE'][1]")
print(unlearning_dataset['TEMPLATE'][1])
"""
def replace_value(example):
    # if example['column_name'] == 'current_value':
    #     example['column_name'] = 'new_value'
    example['TEMPLATE'] = example['TEMPLATE'] + " "
    return example

unlearning_dataset = unlearning_dataset.map(replace_value)
"""
print("modify unlearning_dataset['TEMPLATE'][0]")
print(unlearning_dataset['TEMPLATE'][0])
print("modify unlearning_dataset['TEMPLATE'][1]")
print(unlearning_dataset['TEMPLATE'][1])
"""


# decide which words need to do selected unlearning, need to count perplexity of each input sample
calculate_perplexity_train_dataset = calculate_perplexity_preprocess_dataset(unlearning_dataset, tokenizer)

print("calculate_perplexity_train_dataset")
print(calculate_perplexity_train_dataset)

print("unlearning_dataset")
print(unlearning_dataset)


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

"""
########################## in order to decide which word's index to unlearn, we need to count perplexity first

count_perplexity_batch_size = 8
p_max_length = 1024
p_stride = 1024

### traditional 
calculate_perplexity_collate = functools.partial(calculate_perplexity_collate_fn, pad_index=tokenizer.pad_token_id)
calculate_perplexity_dataloader = torch.utils.data.DataLoader(
    dataset=calculate_perplexity_train_dataset,
    batch_size=count_perplexity_batch_size,
    collate_fn=calculate_perplexity_collate
    )

loss_fct = CrossEntropyLoss(reduction="none")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# print_check_list = ["x33102", "3-9610", "x35275", "x54070", "713-957-7060", "713.345.4727", "5-7857", "3-3985", "x39510", "x36161", "416-352-4580", "3-3210", "39106", "58059"]

print_check_list = ["+583 330853915", "(713)853-9974", "(312) 682-2988", "x53240)", "303-623-0987"]
                    # [8],             [2],             [2],             [4]     , [6]


select_unlearning_index_list = []

# martinc
count_zero_element = 0

with torch.no_grad():
    # batch = {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels, 'remain_attention_mask': remain_attention_mask, 'begin_pii_index': begin_pii_index}
    for batch in tqdm.tqdm(calculate_perplexity_dataloader, desc='evaluating...'):

        global_ids = batch['input_ids'].to(device)
        # length = batch['length']
        global_attention_mask = batch['attention_mask'].to(device)
        global_label = batch['labels'].to(device)
        global_remain_attention_mask = batch['remain_attention_mask'].to(device)
        global_begin_pii_index = batch['begin_pii_index'].to(device)

        seq_len = global_ids.size(1)

        sum_entropy_list = []
        sum_shift_attention_mask_batch_list = []

        prev_end_loc = 0
        # for begin_loc in tqdm(range(0, seq_len, p_stride)):
        for begin_loc in range(0, seq_len, p_stride):
            end_loc = min(begin_loc + p_max_length, seq_len)
            trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
            input_ids = global_ids[:, begin_loc:end_loc]
            attention_mask = global_attention_mask[:, begin_loc:end_loc]
            target_ids = global_label[:, begin_loc:end_loc].clone()
            target_ids[:, :-trg_len] = -100
            remain_attention_mask = global_remain_attention_mask[:, begin_loc:end_loc]


            out_logits = model(input_ids).logits

            shift_logits = out_logits[..., :-1, :].contiguous()
            shift_labels = target_ids[..., 1:].contiguous()
            shift_attention_mask_batch = attention_mask[..., 1:].contiguous()
            shift_remain_attention_mask_batch = remain_attention_mask[..., 1:].contiguous()

            transpose_shift_logits = shift_logits.transpose(1, 2)

            temp_loss = loss_fct(transpose_shift_logits, shift_labels)


            for inner_index in range(temp_loss.shape[0]):
                element_loss = temp_loss[inner_index]
                non_zero_mask = element_loss != 0
                non_zero_element_loss = element_loss[non_zero_mask]

                # count entropy between different labels
                truncate_last_non_zero_element_loss = non_zero_element_loss[:-1]
                shift_non_zero_element_loss = non_zero_element_loss[..., 1:].contiguous()
                entropy_difference_element_loss = truncate_last_non_zero_element_loss - shift_non_zero_element_loss
                entropy_difference_element_loss = entropy_difference_element_loss / truncate_last_non_zero_element_loss

                if(len(entropy_difference_element_loss) >= forget_number_of_token):
                    # max_entropy_diff_index = torch.argmax(entropy_difference_element_loss).item()
                    values, indices = torch.topk(entropy_difference_element_loss, forget_number_of_token)
                    # print("indices")
                    # print(indices)
                    entropy_difference_indices_list = indices.tolist()
                else:
                    # entropy_difference_indices_list = [0]
                    entropy_difference_indices_list =[*range(len(non_zero_element_loss.tolist()))]
                    # martinc test print
                    # print("lower than forget_number_of_token entropy_difference_indices_list")
                    # print(entropy_difference_indices_list)

                # martinc check test
                if (0 in entropy_difference_indices_list):
                    count_zero_element += 1

                select_unlearning_index_list.append(entropy_difference_indices_list)# 6

                
                temp_global_label = global_label[inner_index]
                larger_than_zero_mask = temp_global_label > 0
                larger_than_zero_temp_global_label = temp_global_label[larger_than_zero_mask]

                larger_than_zero_temp_global_label_tokens = tokenizer.convert_ids_to_tokens(larger_than_zero_temp_global_label)
                larger_than_zero_temp_global_label_string = tokenizer.decode(larger_than_zero_temp_global_label)

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

print("len(calculate_perplexity_dataloader):", len(calculate_perplexity_dataloader))
print("count_zero_element:", count_zero_element)
# with no clean space count_zero_element: 119 
# with clean space count_zero_element: 304
# decide to clean space for label, because if we decide to delete "space" it will be meaningless


# insert select_unlearning_index_list into unlearning_dataset
print("unlearning_dataset")
print(unlearning_dataset)

unlearning_dataset = unlearning_dataset.add_column("SELECT_UNLEARNING_INDEX", select_unlearning_index_list)# select_unlearning_index
"""

print("after add column unlearning_dataset")
print(unlearning_dataset)

"""
Dataset({
    features: ['NAME', 'CATEGORY', 'TEMPLATE', 'TEMPLATE_TYPE', 'TARGET', 'TARGET_PII_TYPE', 'SELECT_UNLEARNING_INDEX'],
    num_rows: 815
})
"""

print("fake_dataset")
print(fake_dataset)
"""
Dataset({
    features: ['NAME', 'CATEGORY', 'TEMPLATE', 'TEMPLATE_TYPE', 'TARGET', 'TARGET_PII_TYPE'],
    num_rows: 577
})
"""

# build fake_dataset dictionary 
fake_person_dict = {}
for index in range(len(fake_dataset["NAME"])):
    fake_name = fake_dataset["NAME"][index]
    fake_template = fake_dataset["TEMPLATE"][index]
    fake_target = fake_dataset["TARGET"][index]
    fake_person_dict[fake_name] = {"TEMPLATE": fake_template, "TARGET": fake_target}

print('len(fake_dataset["NAME"]):', len(fake_dataset["NAME"]))# 577
print("len(fake_person_dict):", len(fake_person_dict))# 577


name_accumulate_dict = {}
for index in range(len(unlearning_dataset["NAME"])):
    name = unlearning_dataset["NAME"][index]
    if (name not in name_accumulate_dict):
        name_accumulate_dict[name] = 1
    else:
        name_accumulate_dict[name] += 1


name_copy_accumulate_dict = copy.deepcopy(name_accumulate_dict)

# create combine dataset: unlearning_dataset + fake_dataset

name_list = []
template_list = []
target_list = []
# select_unlearning_index_list = []
fake_name_list = []
fake_template_list = []
fake_target_list = []

# if flag 0, only do gradient ascent on dynamic selected unlearning
# if flag 1, do both gradient ascent on dynamic selected unlearning and gradient descent on fake pii sub
gradient_compensate_flag_list = []

# fake_index = 0
for index in range(len(unlearning_dataset["NAME"])):
    name = unlearning_dataset["NAME"][index]
    name_list.append(name)
    template = unlearning_dataset["TEMPLATE"][index]
    template_list.append(template)
    target = unlearning_dataset["TARGET"][index]
    target_list.append(target)
    # select_unlearning_index = unlearning_dataset["SELECT_UNLEARNING_INDEX"][index]
    # select_unlearning_index_list.append(select_unlearning_index)

    fake_person_information = fake_person_dict[name]
    fake_template = fake_person_information["TEMPLATE"]
    fake_target = fake_person_information["TARGET"]

    fake_template_list.append(fake_template)
    fake_target_list.append(fake_target)

    remain_count = name_copy_accumulate_dict[name]

    if (remain_count == 1):
        gradient_compensate_flag_list.append(1)
    else:
        gradient_compensate_flag_list.append(0)
        
    remain_count -= 1
    name_copy_accumulate_dict[name] = remain_count

    """
    fake_name = fake_dataset["NAME"][fake_index]
    if (fake_name != name):
        fake_index += 1
        # boundary condition: need to break
        # fake_index larger than fake_template_list length
    # fake_name = fake_dataset["NAME"][fake_index]
    # fake_name_list.append(fake_name)
    fake_template = fake_dataset["TEMPLATE"][index]
    fake_template_list.append(fake_template)
    fake_target = fake_dataset["TARGET"][index]
    fake_target_list.append(fake_target)
    """

"""
print("---------------------------------------------------------")
for index in range(len(unlearning_dataset["NAME"])):
    print("name_list[index]:", name_list[index], " gradient_compensate_flag_list[index]:", gradient_compensate_flag_list[index])
    print("template_list[index]")
    print(template_list[index])
    print("target_list[index]")
    print(target_list[index])
    print("select_unlearning_index_list[index]")
    print(select_unlearning_index_list[index])
    print("fake_template_list[index]")
    print(fake_template_list[index])
    print("fake_target_list[index]")
    print(fake_target_list[index])
"""


# whalley_twin_phone_template_dataset
df = pd.DataFrame({
    'NAME': name_list,
    'TEMPLATE': template_list,
    'TARGET': target_list,
    # 'SELECT_UNLEARNING_INDEX': select_unlearning_index_list,
    'FAKE_PII_TEMPLATE': fake_template_list,
    'FAKE_PII_TARGET': fake_target_list,
    'GRADIENT_COMPENSATE_FLAG': gradient_compensate_flag_list
}, columns=['NAME', 'TEMPLATE', 'TARGET', 'FAKE_PII_TEMPLATE', 'FAKE_PII_TARGET', 'GRADIENT_COMPENSATE_FLAG'])# 'SELECT_UNLEARNING_INDEX'

ori_combine_dataset = datasets.Dataset.from_pandas(df)

print("-----------------------------------------------------------------------------------------------------------------")

"""
for index in range(len(ori_combine_dataset['NAME'])):
    name = ori_combine_dataset['NAME'][index]
    template = ori_combine_dataset['TEMPLATE'][index]
    target = ori_combine_dataset['TARGET'][index]
    select_unlearning_index = ori_combine_dataset['SELECT_UNLEARNING_INDEX'][index]
    fake_template = ori_combine_dataset['FAKE_PII_TEMPLATE'][index]
    fake_target = ori_combine_dataset['FAKE_PII_TARGET'][index]
    
    print("next index------------------------:", index)
    print("name")
    print(name)
    print("template")
    print(template)
    print("target")
    print(target)
    print("select_unlearning_index")
    print(select_unlearning_index)
    print("fake_template")
    print(fake_template)
    print("fake_target")
    print(fake_target)
"""

print("ori_combine_dataset")
print(ori_combine_dataset)

"""
combine_dataset
Dataset({
    features: ['NAME', 'TEMPLATE', 'TARGET', 'SELECT_UNLEARNING_INDEX', 'FAKE_TEMPLATE', 'FAKE_TARGET', 'GRADIENT_COMPENSATE_FLAG'],
    num_rows: 815
})
"""

# ori_combine_dataset.save_to_disk(ori_combine_dataset_path)


"""
new_column = [0] * len(unlearning_dataset)
unlearning_dataset = unlearning_dataset.add_column("FAKE_PII_TARGET", new_column)
fake_dataset = fake_dataset.rename_column("NAME", "FAKE_NAME")
fake_dataset = fake_dataset.rename_column("TEMPLATE", "FAKE_PII_TEMPLATE")
fake_dataset = fake_dataset.rename_column("TARGET", "FAKE_PII_TARGET")

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


"""

##### 

# dataset preprocess_dataset
combine_dataset = preprocess_dataset(ori_combine_dataset, tokenizer)
print("preprocess_dataset done")




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
    dataset=combine_dataset,
    batch_size=per_device_train_batch_size,
    collate_fn=collate
    )



# dataloader = torch.utils.data.DataLoader(unlearning_defense_train_dataset, batch_size=per_device_train_batch_size)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model.train()
epoch_losses = []
epoch_accs = []

criterion = CrossEntropyLoss()
criterion = criterion.to(device)

optimizer = AdamW(
    model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(dataloader))  # PyTorch scheduler
# scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(dataloader) * num_train_epochs)  # PyTorch scheduler
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(dataloader) * num_train_epochs * 2)  # PyTorch scheduler


"""
batch = {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels, 
            'fake_input_ids': fake_input_ids, 'fake_attention_mask': fake_attention_mask, 'fake_labels': fake_labels,
            'gradient_compensate_flag': gradient_compensate_flag}
"""

for epoch in range(num_train_epochs):
    # for batch in tqdm.tqdm(dataloader, desc='training...', file=sys.stdout):
    for batch in tqdm.tqdm(dataloader, desc='training...'):
        # dict_keys(['input_ids', 'attention_mask', 'labels', 'fake_input_ids', 'fake_attention_mask', 'fake_labels'])
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        target_ids = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=target_ids)
        loss = outputs.loss

        loss = torch.neg(loss)

        # print("loss---------------------:", loss)

        # accuracy = get_accuracy(prediction, label)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()
        # epoch_losses.append(loss.item())
        # epoch_accs.append(accuracy.item())

        # print("after ascent backward loss:", loss)
        
        """
        ######
        # select data for gradient ascent

        # find which gradient_compensate_flag is one, if one, do gradient descent
        gradient_compensate_flag = batch['gradient_compensate_flag'].to(device)

        non_zero_gradient_compensate_flag = torch.nonzero(gradient_compensate_flag)

        # martinc test print
        print("temp_input_tokens--------------------------------------------------")
        for check_index in range(len(input_ids)):
            temp_input_tokens = tokenizer.convert_ids_to_tokens(input_ids[check_index])
            print(temp_input_tokens)

        if (len(non_zero_gradient_compensate_flag) > 0):

            gradient_ascent_selected_index = torch.squeeze(non_zero_gradient_compensate_flag)

            lr_rate = scheduler.get_last_lr()

            fake_input_ids = batch['fake_input_ids'].to(device)
            fake_attention_mask = batch['fake_attention_mask'].to(device)
            fake_labels = batch['fake_labels'].to(device)

            fake_input_ids = torch.index_select(fake_input_ids, 0, gradient_ascent_selected_index)
            fake_attention_mask = torch.index_select(fake_attention_mask, 0, gradient_ascent_selected_index)
            fake_labels = torch.index_select(fake_labels, 0, gradient_ascent_selected_index)

            # martinc test print
            print("temp_fake_input_tokens--------------------------------------------------")
            for check_index in range(len(fake_input_ids)):
                temp_fake_input_tokens = tokenizer.convert_ids_to_tokens(fake_input_ids[check_index])
                print(temp_fake_input_tokens)


            outputs = model(fake_input_ids, attention_mask=fake_attention_mask, labels=fake_labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()

            # print("lr_rate:", lr_rate, " after descent backward loss:", loss)
        """
        print("loss---------------------:", loss)

# model.save_pretrained("/home/my_name/Desktop/t5small", from_pt=True) 

print("before save model")
# ../selected_unlearning_max_and_fake_pii_min_dir/test/
# save_model_path = "../selected_unlearning_max_and_fake_pii_min_dir/test/min_max_test"
# save_model_path = "../selected_unlearning_max_and_fake_pii_min_dir/softprompt_adv_train_dir/adv_epoch_0/e_1_defense_fake_pii"
model.save_pretrained(output_filepath, from_pt=True)
print("done save model output_filepath:", output_filepath)