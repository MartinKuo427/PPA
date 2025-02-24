#!/bin/bash

# Check if the correct number of arguments are passed
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <attack_dataset_path> <model_name> <softprompt> <output_filepath>"
    exit 1
fi

# Assign input arguments to variables for better readability
ATTACK_DATASET_PATH=$1
MODEL_NAME=$2
SOFTPROMPT=$3
OUTPUT_FILEPATH=$4


# Execute the first Python script for attacking the phone data
python3 address_aeslc_propile_softprompt_attack.py --attack_dataset_path $ATTACK_DATASET_PATH --model_name $MODEL_NAME --peft_model_id $SOFTPROMPT --output_filepath $OUTPUT_FILEPATH

# Execute the second Python script to integrate and process attack results
python3 address_aeslc_integrate_diff_propile_attack_results.py --rootdir $OUTPUT_FILEPATH --output_filepath "${OUTPUT_FILEPATH}/softprompt_attack_results.csv"

# Execute the third Python script to evaluate scores from the attack results
python3 address_aeslc_propile_score_eval.py --gather_attack_data_path "${OUTPUT_FILEPATH}/softprompt_attack_results.csv" --output_filepath "${OUTPUT_FILEPATH}/softprompt_attack_score.csv"
