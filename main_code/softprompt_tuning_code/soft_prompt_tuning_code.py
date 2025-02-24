import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
)
import transformers
import torch
import numpy as np
import datasets
from typing import Any, Dict, List
import contextlib
from transformers import Trainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from peft import get_peft_model, TaskType, PromptTuningConfig, PromptTuningInit
import argparse

set_seed(0)  # for reproducibility
# Set the random seed for NumPy
np.random.seed(0)
# Set the random seed for PyTorch
torch.manual_seed(0)
# If you are using CUDA (i.e., a GPU), also set the seed for it
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True

transformers.logging.set_verbosity_error()

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

def preprocess_dataset(
    dataset,
    tokenizer
):

    column_names = list(next(iter(dataset)).keys())

    def preprocess_select_template_target(examples):
        for index in range(len(examples['TEMPLATE'])):
            temp_template = examples['TEMPLATE'][index]
            temp_target = examples['TARGET'][index]
            examples['TEMPLATE'][index] = [temp_template[0]]
            examples['TARGET'][index] = [temp_target[0]]

        return examples

    def preprocess_pretrain_dataset(examples: Dict[str, List[Any]]) -> Dict[str, Any]:
        # build grouped texts with format `X1 X2 X3 ...`
        # flatten examples["TEMPLATE"]
        flatten_template_list = []
        for index in range(len(examples["TEMPLATE"])):
            flatten_template_list.append(examples["TEMPLATE"][index][0])

        # text_target
        tokenized_examples = tokenizer(flatten_template_list, text_target=flatten_template_list)# text_target column name labels

        # flatten examples["TARGET"]
        flatten_target_list = []
        for index in range(len(examples["TARGET"])):
            # add eos token at the last position of every response
            flatten_target_list.append(examples["TARGET"][index][0])

        kwargs_response = dict(add_special_tokens=False)
        response = tokenizer(flatten_target_list, **kwargs_response)

        # assign new input and new label: combine input_ids and response
        for index in range(len(tokenized_examples["input_ids"])):
            # input_ids
            temp_tokenized_examples_input_ids = tokenized_examples["input_ids"][index]
            temp_response_input_ids = response["input_ids"][index]

            convert_temp_response_input_ids_tokens = tokenizer.convert_ids_to_tokens(temp_response_input_ids)
            
            # we need to remove additional space tokens
            if (convert_temp_response_input_ids_tokens[0] == '▁'):
                temp_response_input_ids = temp_response_input_ids[1:]
                response["attention_mask"][index] = response["attention_mask"][index][1:]

            temp_combine_tokenized_examples_input_ids = temp_tokenized_examples_input_ids + temp_response_input_ids

            temp_tokenized_examples_attention_mask = tokenized_examples["attention_mask"][index]
            temp_response_attention_mask = response["attention_mask"][index]
            temp_combine_tokenized_examples_attention_mask = temp_tokenized_examples_attention_mask + temp_response_attention_mask
            temp_labels = [-100] * len(temp_tokenized_examples_input_ids) + temp_response_input_ids
            
            MAX_LENGTH = 256
            
            if (len(temp_combine_tokenized_examples_input_ids) > MAX_LENGTH):
                temp_combine_tokenized_examples_input_ids = temp_combine_tokenized_examples_input_ids[:MAX_LENGTH]
                temp_combine_tokenized_examples_attention_mask = temp_combine_tokenized_examples_attention_mask[:MAX_LENGTH]
                temp_labels = temp_labels[:MAX_LENGTH]

            tokenized_examples["input_ids"][index] = temp_combine_tokenized_examples_input_ids
            tokenized_examples["attention_mask"][index] = temp_combine_tokenized_examples_attention_mask
            tokenized_examples["labels"][index] = temp_labels

        return tokenized_examples

    dataset = dataset.filter(lambda example: example["TEMPLATE"])

    preprocess_select_func = preprocess_select_template_target
    preprocess_func = preprocess_pretrain_dataset

    with main_process_first():
        kwargs = {}
        num_threads = get_num_workers()
        # if not data_args.streaming:
        kwargs = dict(
            num_proc=num_threads if num_threads > 0 else None,
            load_from_cache_file=False,
            desc="Running tokenizer on dataset"
        )

        dataset = dataset.map(
            preprocess_select_func,
            batched=True,
            **kwargs
        )

        dataset = dataset.map(
            preprocess_func,
            batched=True,            
            remove_columns=column_names,
            **kwargs
        )
        
    return dataset


parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="", help="model_name")
parser.add_argument("--tokenizer_name", type=str, default="meta-llama/Llama-2-7b-hf", help="tokenizer_name")
parser.add_argument("--train_dataset_path", type=str, default="./data/", help="train_dataset_path")
parser.add_argument("--valid_dataset_path", type=str, default="./data/", help="valid_dataset_path")
parser.add_argument("--prompt_tuning_init_text", type=str, default="", help="prompt_tuning_init_text")
parser.add_argument("--num_virtual_tokens", type=int, default=20)
parser.add_argument("--num_train_epochs", type=int, default=20)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--learning_rate", type=float, default=5e-3)
parser.add_argument("--optim", type=str, default="paged_adamw_32bit")
parser.add_argument("--warmup_ratio", type=float, default=0.0)
parser.add_argument("--logging_steps", type=int, default=1)
parser.add_argument("--output_dir", type=str, default="./data/", help="output directory")

args = parser.parse_args()

model_name = args.model_name
tokenizer_name = args.tokenizer_name
train_dataset_path = args.train_dataset_path
valid_dataset_path = args.valid_dataset_path
prompt_tuning_init_text = args.prompt_tuning_init_text
num_virtual_tokens = args.num_virtual_tokens
num_train_epochs = args.num_train_epochs
batch_size = args.batch_size
learning_rate = args.learning_rate
optim = args.optim
warmup_ratio = args.warmup_ratio
logging_steps = args.logging_steps
output_dir = args.output_dir

print("train_dataset_path:", train_dataset_path)
print("valid_dataset_path:", valid_dataset_path)
print("num_train_epochs:", num_train_epochs)
print("prompt_tuning_init_text:", prompt_tuning_init_text)
print("num_virtual_tokens:", num_virtual_tokens)
print("output_dir:", output_dir)

train_dataset = datasets.load_from_disk(train_dataset_path)

print("train_dataset")
print(train_dataset)

train_dataset = train_dataset.shuffle(seed=42)

eval_dataset = datasets.load_from_disk(valid_dataset_path)

print("eval_dataset")
print(eval_dataset)

# Load LLaMA tokenizer
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training

# dataset preprocess_dataset
train_dataset = preprocess_dataset(train_dataset, tokenizer)

# eval_dataset preprocess_dataset
eval_dataset = preprocess_dataset(eval_dataset, tokenizer)

print("after train_dataset preprocess_dataset")
print(train_dataset)

print("after eval_dataset preprocess_dataset")
print(eval_dataset)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto"
)
model.config.use_cache = False
model.config.pretraining_tp = 1

our_tokenizer_name = model_name
# initialize pii_type Soft Prompt
config = PromptTuningConfig(
    peft_type="PROMPT_TUNING",
    task_type=TaskType.CAUSAL_LM,
    prompt_tuning_init=PromptTuningInit.TEXT,
    prompt_tuning_init_text=prompt_tuning_init_text,# "phone" # "address"
    num_virtual_tokens=num_virtual_tokens,
    # tokenizer_kwargs=tokenizer
    tokenizer_name_or_path = tokenizer_name)

model = get_peft_model(model, config) # 生成Prompt-Tuning对应的model
print(model.print_trainable_parameters())


data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, return_tensors="pt")
 
# Enable fp16/bf16 training (set bf16 to True with an A100)
fp16 = False
bf16 = False
 
# Batch size per GPU for training
per_device_train_batch_size = batch_size
# Batch size per GPU for evaluation
per_device_eval_batch_size = batch_size
 
# Number of training steps (overrides num_train_epochs)
max_steps = -1
 
# Group sequences into batches with same length
# Saves memory and speeds up training considerably
group_by_length = True

# Set training parameters
training_arguments = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    per_device_eval_batch_size=per_device_eval_batch_size,
    optim=optim,
    save_strategy="epoch",
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    fp16=fp16,
    bf16=bf16,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=group_by_length,
    disable_tqdm=False,
    save_total_limit=1,
    load_best_model_at_end = True,
    evaluation_strategy='epoch'     # "no": No evaluation is done during training.
                                    # "steps": Evaluation is done (and logged) every steps
                                    # "epoch": Evaluation is done at the end of each epoch.
)



print("train_dataset--------------------------------")
print(train_dataset)

# Initialize our Trainer
trainer = Trainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=training_arguments,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()