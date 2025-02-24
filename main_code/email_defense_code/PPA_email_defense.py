import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
    AdamW,
    get_cosine_schedule_with_warmup,
)
import transformers
import torch
import numpy as np
import pandas as pd
import datasets
from typing import Any, Dict, List
import contextlib
import tqdm
import argparse
from torch.nn import CrossEntropyLoss
import copy
import functools

transformers.logging.set_verbosity_error()

set_seed(0)  # for reproducibility
# Set the random seed for NumPy
np.random.seed(0)
# Set the random seed for PyTorch
torch.manual_seed(0)
# If you are using CUDA (i.e., a GPU), also set the seed for it
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True

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
    return 1

def calculate_perplexity_preprocess_dataset(
    dataset,
    tokenizer
):

    column_names = list(next(iter(dataset)).keys())

    def preprocess_pretrain_dataset(examples: Dict[str, List[Any]]) -> Dict[str, Any]:
        # build grouped texts with format `X1 X2 X3 ...`
        max_length = 34
        template_tokenized_examples = tokenizer(examples["TEMPLATE"], text_target=examples["TEMPLATE"], max_length=max_length, truncation=True)
        target_tokenized_examples = tokenizer(examples["TARGET"], text_target=examples["TARGET"], max_length=max_length, truncation=True, add_special_tokens=False)
        
        # when llamatokenizer tokenize labels, it may add space in the first position of the label, so we need to remove it, 
        # if there is a "space" at the begining of the labels, we will remove it.
        space_id = tokenizer.convert_tokens_to_ids('▁')

        for index in range(len(target_tokenized_examples["labels"])):
            temp_target_input_ids = target_tokenized_examples["input_ids"][index]
            temp_target_attention_mask = target_tokenized_examples["attention_mask"][index]
            temp_target_labels = target_tokenized_examples["labels"][index]
            if (temp_target_input_ids[0] == space_id):
                temp_target_input_ids = temp_target_input_ids[1:]
                temp_target_attention_mask = temp_target_attention_mask[1:]
                temp_target_labels = temp_target_labels[1:]
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
            remain_concatenate_attention_mask[:len(temp_template_attention_mask)] = 0
            remain_concatenate_attention_mask = list(remain_concatenate_attention_mask)

            template_tokenized_examples["input_ids"][index] = concatenate_input_ids
            template_tokenized_examples["attention_mask"][index] = concatenate_attention_mask
            template_tokenized_examples["labels"][index] = concatenate_labels
            template_tokenized_examples["remain_attention_mask"][index] = remain_concatenate_attention_mask
            template_tokenized_examples["begin_pii_index"][index] = len(temp_template_input_ids)

        return template_tokenized_examples
    
    preprocess_func = preprocess_pretrain_dataset
    with main_process_first():
        kwargs = {}
        num_threads = get_num_workers()

        kwargs = dict(
            num_proc=num_threads if num_threads > 0 else None,
            load_from_cache_file=False,
            desc="Running tokenizer on dataset"
        )

        dataset = dataset.map(
            preprocess_func,
            batched=True,            
            remove_columns=column_names,
            **kwargs
        )
    
    return dataset


def preprocess_dataset(
    ori_combine_dataset,
    tokenizer
):
    # features: ['NAME', 'TEMPLATE', 'TARGET', 'SELECT_UNLEARNING_INDEX', 'FAKE_TEMPLATE', 'FAKE_TARGET', 'GRADIENT_COMPENSATE_FLAG']
    
    def preprocess_pretrain_dataset(examples: Dict[str, List[Any]]) -> Dict[str, Any]:
        # build grouped texts with format `X1 X2 X3 ...`
        select_unlearning_index_list = examples["SELECT_UNLEARNING_INDEX"]
        max_length = 34

        template_tokenized_examples = tokenizer(examples["TEMPLATE"], text_target=examples["TEMPLATE"], max_length=max_length, truncation=True)
        target_tokenized_examples = tokenizer(examples["TARGET"], text_target=examples["TARGET"], max_length=max_length, truncation=True, add_special_tokens=False)
        
        # handle unlearning dataset tokenized_examples-----------------------------------------------------------------------
        # when llamatokenizer tokenize labels, it may add space in the first position of the label, so we need to remove it, 
        # if there is a "space" at the begining of the labels, we will remove it.
        space_id = tokenizer.convert_tokens_to_ids('▁')

        for index in range(len(target_tokenized_examples["labels"])):
            temp_target_input_ids = target_tokenized_examples["input_ids"][index]
            temp_target_attention_mask = target_tokenized_examples["attention_mask"][index]
            temp_target_labels = target_tokenized_examples["labels"][index]
            if (temp_target_input_ids[0] == space_id):
                temp_target_input_ids = temp_target_input_ids[1:]
                temp_target_attention_mask = temp_target_attention_mask[1:]
                temp_target_labels = temp_target_labels[1:]
                target_tokenized_examples["input_ids"][index] = temp_target_input_ids
                target_tokenized_examples["attention_mask"][index] = temp_target_attention_mask
                target_tokenized_examples["labels"][index] = temp_target_labels

        ignore_index = -100
        for index in range(len(target_tokenized_examples["labels"])):
            hope_designed_forget_index = select_unlearning_index_list[index]
            temp_target_labels = target_tokenized_examples["labels"][index]
            labels_to_tokens = tokenizer.convert_ids_to_tokens(temp_target_labels)
            modified_temp_target_labels = [ignore_index] * len(temp_target_labels)
            designed_forget_index = hope_designed_forget_index

            for target_index in range(len(labels_to_tokens)):
                if (target_index in designed_forget_index):
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

        max_length = 34
        fake_template_tokenized_examples = tokenizer(examples["FAKE_PII_TEMPLATE"], text_target=examples["FAKE_PII_TEMPLATE"], max_length=max_length, truncation=True)
        fake_target_tokenized_examples = tokenizer(examples["FAKE_PII_TARGET"], text_target=examples["FAKE_PII_TARGET"], max_length=max_length, truncation=True, add_special_tokens=False)

        # handle fake pii sub dataset tokenized_examples
        ignore_index = -100
        for index in range(len(fake_target_tokenized_examples["labels"])):
            temp_target_labels = fake_target_tokenized_examples["labels"][index]
            labels_to_tokens = tokenizer.convert_ids_to_tokens(temp_target_labels)
            modified_temp_target_labels = [ignore_index] * len(temp_target_labels)

            for target_index in range(len(labels_to_tokens)):
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

        fake_template_tokenized_examples['fake_input_ids'] = fake_template_tokenized_examples['input_ids']
        del fake_template_tokenized_examples['input_ids']

        fake_template_tokenized_examples['fake_attention_mask'] = fake_template_tokenized_examples['attention_mask']
        del fake_template_tokenized_examples['attention_mask']

        fake_template_tokenized_examples['fake_labels'] = fake_template_tokenized_examples['labels']
        del fake_template_tokenized_examples['labels']

        template_tokenized_examples.update(fake_template_tokenized_examples)

        template_tokenized_examples['gradient_compensate_flag'] = examples["GRADIENT_COMPENSATE_FLAG"]

        return template_tokenized_examples

    preprocess_pretrain_func = preprocess_pretrain_dataset
    with main_process_first():
        kwargs = {}
        num_threads = get_num_workers()
        kwargs = dict(
            num_proc=num_threads if num_threads > 0 else None,
            load_from_cache_file=False,
            desc="Running tokenizer on dataset"
        )

        column_names = list(next(iter(ori_combine_dataset)).keys())

        combine_dataset = ori_combine_dataset.map(
            preprocess_pretrain_func,
            batched=True,            
            remove_columns=column_names,
            **kwargs
        )

    return combine_dataset

def collate(batch, pad_index):
    input_ids = [torch.LongTensor(i['input_ids']) for i in batch]
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, padding_value=pad_index, batch_first=True)
    
    attention_mask = [torch.LongTensor(i['attention_mask']) for i in batch]
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, padding_value=0, batch_first=True)
    
    labels = [torch.LongTensor(i['labels']) for i in batch]
    labels = torch.nn.utils.rnn.pad_sequence(labels, padding_value=-100, batch_first=True)


    fake_input_ids = [torch.LongTensor(i['fake_input_ids']) for i in batch]
    fake_input_ids = torch.nn.utils.rnn.pad_sequence(fake_input_ids, padding_value=pad_index, batch_first=True)
    
    fake_attention_mask = [torch.LongTensor(i['fake_attention_mask']) for i in batch]
    fake_attention_mask = torch.nn.utils.rnn.pad_sequence(fake_attention_mask, padding_value=0, batch_first=True)
    
    fake_labels = [torch.LongTensor(i['fake_labels']) for i in batch]
    fake_labels = torch.nn.utils.rnn.pad_sequence(fake_labels, padding_value=-100, batch_first=True)

    gradient_compensate_flag = torch.LongTensor([i['gradient_compensate_flag'] for i in batch])

    batch = {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels, 
             'fake_input_ids': fake_input_ids, 'fake_attention_mask': fake_attention_mask, 'fake_labels': fake_labels,
             'gradient_compensate_flag': gradient_compensate_flag}
    return batch

collate_fn = collate

def calculate_perplexity_collate(batch, pad_index):
    # dict_keys(['input_ids', 'attention_mask', 'labels', 'fake_input_ids', 'fake_attention_mask', 'fake_labels'])

    input_ids = [torch.LongTensor(i['input_ids']) for i in batch]
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, padding_value=pad_index, batch_first=True)
    
    attention_mask = [torch.LongTensor(i['attention_mask']) for i in batch]
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, padding_value=0, batch_first=True)
    
    labels = [torch.LongTensor(i['labels']) for i in batch]
    labels = torch.nn.utils.rnn.pad_sequence(labels, padding_value=-100, batch_first=True)

    remain_attention_mask = [torch.LongTensor(i['remain_attention_mask']) for i in batch]
    remain_attention_mask = torch.nn.utils.rnn.pad_sequence(remain_attention_mask, padding_value=0, batch_first=True)

    begin_pii_index = torch.LongTensor([i['begin_pii_index'] for i in batch])

    batch = {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels, 'remain_attention_mask': remain_attention_mask, 'begin_pii_index': begin_pii_index}
    return batch

calculate_perplexity_collate_fn = calculate_perplexity_collate

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="", help="model_name")
    parser.add_argument("--tokenizer_name", type=str, default="meta-llama/Llama-2-7b-hf", help="tokenizer_name")
    parser.add_argument("--unlearning_data_path", type=str, default="", help="unlearning_data_path")
    parser.add_argument("--fake_data_path", type=str, default="", help="fake_data_path")
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("-f", "--forget_number_of_token", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_grad_norm", type=float, default=0.3)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--output_filepath", type=str, default="./data/", help="output directory")
    args = parser.parse_args()

    model_name = args.model_name
    print("model_name:", model_name)

    tokenizer_name = args.tokenizer_name

    forget_number_of_token = args.forget_number_of_token
    print("forget_number_of_token:", forget_number_of_token)

    unlearning_data_path = args.unlearning_data_path
    print("unlearning_data_path:", unlearning_data_path)

    fake_data_path = args.fake_data_path
    print("fake_data_path:", fake_data_path)

    output_filepath = args.output_filepath
    print("output_filepath:", output_filepath)

    num_train_epochs = args.num_train_epochs
    batch_size= args.batch_size
    max_grad_norm = args.max_grad_norm
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    warmup_ratio = args.warmup_ratio

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)# clean_up_tokenization_spaces decode_with_prefix_space=False
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training

    unlearning_dataset = datasets.load_from_disk(unlearning_data_path)

    print("unlearning_dataset")
    print(unlearning_dataset)

    fake_dataset = datasets.load_from_disk(fake_data_path)

    print("fake_dataset")
    print(fake_dataset)

    def replace_value(example):
        example['TEMPLATE'] = example['TEMPLATE'] + " "
        return example

    unlearning_dataset = unlearning_dataset.map(replace_value)

    # decide which words need to do selected unlearning, need to count perplexity of each input sample
    calculate_perplexity_train_dataset = calculate_perplexity_preprocess_dataset(unlearning_dataset, tokenizer)

    print("calculate_perplexity_train_dataset")
    print(calculate_perplexity_train_dataset)

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto"
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

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

    select_unlearning_index_list = []

    with torch.no_grad():
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
                trg_len = end_loc - prev_end_loc
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
                        values, indices = torch.topk(entropy_difference_element_loss, forget_number_of_token)
                        entropy_difference_indices_list = indices.tolist()
                    else:
                        entropy_difference_indices_list =[*range(len(non_zero_element_loss.tolist()))]

                    select_unlearning_index_list.append(entropy_difference_indices_list)

                    temp_global_label = global_label[inner_index]
                    larger_than_zero_mask = temp_global_label > 0
                    larger_than_zero_temp_global_label = temp_global_label[larger_than_zero_mask]

                    larger_than_zero_temp_global_label_tokens = tokenizer.convert_ids_to_tokens(larger_than_zero_temp_global_label)
                    larger_than_zero_temp_global_label_string = tokenizer.decode(larger_than_zero_temp_global_label)

                prev_end_loc = end_loc
                if end_loc == seq_len:
                    break

    unlearning_dataset = unlearning_dataset.add_column("SELECT_UNLEARNING_INDEX", select_unlearning_index_list)# select_unlearning_index

    # build fake_dataset dictionary 
    fake_person_dict = {}
    for index in range(len(fake_dataset["NAME"])):
        fake_name = fake_dataset["NAME"][index]
        fake_template = fake_dataset["TEMPLATE"][index]
        fake_target = fake_dataset["TARGET"][index]
        fake_person_dict[fake_name] = {"TEMPLATE": fake_template, "TARGET": fake_target}

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
    select_unlearning_index_list = []
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
        select_unlearning_index = unlearning_dataset["SELECT_UNLEARNING_INDEX"][index]
        select_unlearning_index_list.append(select_unlearning_index)

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

    df = pd.DataFrame({
        'NAME': name_list,
        'TEMPLATE': template_list,
        'TARGET': target_list,
        'SELECT_UNLEARNING_INDEX': select_unlearning_index_list,
        'FAKE_PII_TEMPLATE': fake_template_list,
        'FAKE_PII_TARGET': fake_target_list,
        'GRADIENT_COMPENSATE_FLAG': gradient_compensate_flag_list
    }, columns=['NAME', 'TEMPLATE', 'TARGET', 'SELECT_UNLEARNING_INDEX', 'FAKE_PII_TEMPLATE', 'FAKE_PII_TARGET', 'GRADIENT_COMPENSATE_FLAG'])

    ori_combine_dataset = datasets.Dataset.from_pandas(df)

    print("ori_combine_dataset")
    print(ori_combine_dataset)

    # dataset preprocess_dataset
    combine_dataset = preprocess_dataset(ori_combine_dataset, tokenizer)
    
    # Enable fp16/bf16 training (set bf16 to True with an A100)
    fp16 = False
    bf16 = False
    
    # Batch size per GPU for training
    per_device_train_batch_size = batch_size
    
    # Batch size per GPU for evaluation
    per_device_eval_batch_size = batch_size

    collate = functools.partial(collate_fn, pad_index=tokenizer.pad_token_id)
    dataloader = torch.utils.data.DataLoader(
        dataset=combine_dataset,
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

    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(dataloader) * num_train_epochs * 2)  # PyTorch scheduler

    for epoch in range(num_train_epochs):
        for batch in tqdm.tqdm(dataloader, desc='training...'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            target_ids = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=target_ids)
            loss = outputs.loss

            loss = torch.neg(loss)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            # find which gradient_compensate_flag is one, if one, do gradient descent
            gradient_compensate_flag = batch['gradient_compensate_flag'].to(device)

            non_zero_gradient_compensate_flag = torch.nonzero(gradient_compensate_flag)

            if (len(non_zero_gradient_compensate_flag) > 0):
                gradient_ascent_selected_index = torch.squeeze(non_zero_gradient_compensate_flag)

                fake_input_ids = batch['fake_input_ids'].to(device)
                fake_attention_mask = batch['fake_attention_mask'].to(device)
                fake_labels = batch['fake_labels'].to(device)

                fake_input_ids = torch.index_select(fake_input_ids, 0, gradient_ascent_selected_index)
                fake_attention_mask = torch.index_select(fake_attention_mask, 0, gradient_ascent_selected_index)
                fake_labels = torch.index_select(fake_labels, 0, gradient_ascent_selected_index)

                outputs = model(fake_input_ids, attention_mask=fake_attention_mask, labels=fake_labels)
                loss = outputs.loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()

    model.save_pretrained(output_filepath, from_pt=True)
    print("done save model output_filepath:", output_filepath)