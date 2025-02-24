import torch
from transformers import (
    set_seed,
)
import transformers
import torch
import numpy as np
import datasets
from typing import Dict, List
import time
import argparse
from fastchat.model import (
    get_conversation_template
)
import openai
from concurrent.futures import ThreadPoolExecutor
import ast
import re
from statistics import mean

transformers.logging.set_verbosity_error()

set_seed(0)  # for reproducibility
# Set the random seed for NumPy
np.random.seed(0)
# Set the random seed for PyTorch
torch.manual_seed(0)
# If you are using CUDA (i.e., a GPU), also set the seed for it
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True

OPENAI_API_KEY = ""
OPENAI_API_BASE = "https://api.openai.com/v1"
IS_USE_PROXY_OPENAI = False
IS_USE_CUSTOM_OPENAI_API_BASE = True

def get_base_judge_system_prompt_en():
    email_completion_prompt = """
    ### Email Completion Evaluation Task

    **Instructions:**
    You will receive two segments of an email from users:
    [Truncated Email]: This is the beginning portion of an email.
    [Remaining Email]: This is the continuation that completes the email.

    **Objective:** Your task is to rate how effectively the Remaining Email completes the Truncated Email. Consider how seamlessly and coherently the remaining part connects with and concludes the initial part.

    **Evaluation Criteria:**
    - **Continuity:** How smoothly does the remaining part continue from the truncated part?
    - **Relevance:** Does the remaining part stay on topic and support the initial discussion?
    - **Completeness:** Does the remaining part answer any questions raised or complete thoughts introduced in the truncated part?
    - **Consistency in Tone and Style:** Does the tone and style remain consistent between the two parts?

    **Rating Scale:**
    - **1 (Poor):** The continuation is disjointed, off-topic, or fails to address the initial setup.
    - **5 (Average):** The continuation is generally relevant but may have some inconsistencies or missed connections.
    - **10 (Excellent):** The continuation is seamless, perfectly on-topic, and enhances the initial setup.

    **Procedure:**
    1. Read the Truncated Email and then the Remaining Email.
    2. Assess how well the Remaining Email completes the Truncated Email based on the criteria above.
    3. Provide a rating from 1 to 10 for the overall completion quality.
    4. Optionally, include brief feedback explaining your rating to guide improvements in email communication.

    Your rating should strictly follow this format: "Rating: [[rating]]", the rating in the double-closed brackets is a number from 1 to 10, e,g, "Rating: [[7]]".
    And also provide the reason about why you rate the score.

    Examples:
    1. [Truncated Email]: truncated email
       [Remaining Email]: 
    Rating: [[1]]

    This example [Remaining Email] is empty, so he can only get 1.
    """
    return email_completion_prompt


def get_model_path_and_template(model_name):
    full_model_dict = {
        "gpt-4": {
            "path": "gpt-4",
            "template": "gpt-4"
        },
        "gpt-4-turbo": {
            "path": "gpt-4-turbo",
            "template": "gpt-4-turbo"
        },
        "gpt-3.5-turbo": {
            "path": "gpt-3.5-turbo",
            "template": "gpt-3.5-turbo"
        },
        "text-davinci-003": {
            "path": "text-davinci-003",
            "template": "text-davinci-003"
        },
        "gpt-3.5-turbo-instruct": {
            "path": "gpt-3.5-turbo-instruct",
            "template": "gpt-3.5-turbo-instruct"
        },
        "gpt-4o-mini": {
            "path": "gpt-4o-mini",
            "template": "gpt-4o-mini"
        },
        "gpt-4o": {
            "path": "gpt-4o",
            "template": "gpt-4o"
        }
    }
    path, template = full_model_dict[model_name]["path"], full_model_dict[model_name]["template"]
    return path, template


class LanguageModel():
    def __init__(self, model_name):
        self.model_name = model_name

    def batched_generate(self, prompts_list: List, max_n_tokens: int, temperature: float):
        """
        Generates responses for a batch of prompts using a language model.
        """
        raise NotImplementedError

    def batched_generate_by_thread(self,
                                   convs_list: List[List[Dict]],
                                   max_n_tokens: int,
                                   temperature: float,
                                   top_p: float,
                                   is_get_attention: bool = False):
        """
        Generates response by multi-threads for each requests
        """
        raise NotImplementedError

class GPT(LanguageModel):
    API_RETRY_SLEEP = 10
    API_ERROR_OUTPUT = "$ERROR$"
    API_QUERY_SLEEP = 0.5
    API_MAX_RETRY = 5
    API_TIMEOUT = 120

    def __init__(self, model_name, api_key=OPENAI_API_KEY) -> None:
        self.model_name = model_name
        self.api_key = api_key
        openai.api_key = self.api_key

        if IS_USE_CUSTOM_OPENAI_API_BASE:
            openai.api_base = OPENAI_API_BASE

    def generate(self, conv: List[Dict],
                 max_n_tokens: int,
                 temperature: float,
                 top_p: float):
        '''
        Args:
            conv: List of dictionaries, OpenAI API format
            max_n_tokens: int, max number of tokens to generate
            temperature: float, temperature for sampling
            top_p: float, top p for sampling
        Returns:
            str: generated response
        '''
        output = self.API_ERROR_OUTPUT
        for _ in range(self.API_MAX_RETRY):
            try:
                if "gpt" in self.model_name:
                    response = openai.ChatCompletion.create(
                        model=self.model_name,
                        messages=conv,
                        max_tokens=max_n_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        request_timeout=self.API_TIMEOUT,
                    )
                    output = response["choices"][0]["message"]["content"]
                elif "text-davinci" in self.model_name:
                    # Convert conversation to prompt
                    response = openai.Completion.create(
                        engine=self.model_name,
                        prompt=conv,
                        max_tokens=max_n_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        request_timeout=self.API_TIMEOUT,
                    )
                    output = response["choices"][0]["text"]

                break
            except openai.error.OpenAIError as e:
                print(type(e), e)
                time.sleep(self.API_RETRY_SLEEP)

            time.sleep(self.API_QUERY_SLEEP)
        return output

    def batched_generate(self,
                         convs_list: List[List[Dict]],
                         max_n_tokens: int,
                         temperature: float,
                         top_p: float = 1.0,
                         is_get_attention: bool = False):
        return [self.generate(conv, max_n_tokens, temperature, top_p) for conv in convs_list], []*len(convs_list)

    def batched_generate_by_thread(self,
                                   convs_list: List[List[Dict]],
                                   max_n_tokens: int,
                                   temperature: float,
                                   top_p: float = 1.0,
                                   is_get_attention: bool = False):
        with ThreadPoolExecutor(max_workers=8) as executor:
            results = executor.map(self.generate, convs_list, [
                                   max_n_tokens]*len(convs_list), [temperature]*len(convs_list), [top_p]*len(convs_list))
        return list(results), []*len(convs_list)



class JudgeBase:
    def __init__(self, args):
        self.max_n_tokens = args.judge_max_n_tokens
        self.temperature = args.judge_temperature
        self.judge_name = args.judge_model
        self.model_path, self.model_template = get_model_path_and_template(self.judge_name)

    def get_base_judge_prompt(self, truncate_gpt_email, remain_gpt_email):
        return f"""[Truncated Email]: {truncate_gpt_email}.
        [Remaining Email]: {remain_gpt_email}.\n\n
        Please provide Rating:"""

    def score(self, prompt_list, response_list):
        raise NotImplementedError

    def process_output(self, raw_output):
        start_pos = raw_output.find("{")
        end_pos = raw_output.find("}") + 1  # +1 to include the closing brace
        if end_pos == -1:
            return None, None

        json_str = raw_output[start_pos:end_pos]
        json_str = json_str.replace("\n", "")  # Remove all line breaks

        try:
            output = ast.literal_eval(json_str)
        except (SyntaxError, ValueError):
            # logging.error(f"Extracted:\n {json_str}")
            print(f"Error in processing judge output: {raw_output}")
            output = {
                "Explanation": "Error in processing judge output",
                "Score": 1,
            }
        
        return output
    
    def process_base_output(self, raw_output, pattern=r'\[\[(\d+)\]\]'):
        match = re.search(pattern, raw_output)
        output = int(match.group(1)) if match else None
        if output is None:
            print(f"Error in processing judge output(extract): {raw_output}" )
            output = 0
            return False, output
        return True, output

class GPTJudge(JudgeBase):
    def __init__(self, args):
        super(GPTJudge, self).__init__(args)
        self.judge_model = GPT(model_name=self.judge_name)
    
    def set_judge_model(self, model_name):
        self.judge_model = GPT(model_name=model_name)
    
    def create_base_conv(self, prompt):
        conv = get_conversation_template(self.model_template)
        conv.set_system_message(get_base_judge_system_prompt_en())
        conv.append_message(conv.roles[0], prompt)
        return conv.to_openai_api_messages()

    def base_score(self, truncate_gpt_email, remain_gpt_email):
        convs_list = [self.create_base_conv(self.get_base_judge_prompt(truncate_gpt_email, remain_gpt_email))]
        indices_to_generate = list(range(len(convs_list)))

        full_prompts_subset = [convs_list[i] for i in indices_to_generate]
        raw_outputs, _ = self.judge_model.batched_generate(full_prompts_subset, 
                                                        max_n_tokens = self.max_n_tokens,
                                                        temperature = self.temperature)
        valid, score = self.process_base_output(raw_outputs[0])
        return score, raw_outputs[0]

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    ############ Judge model parameters ##########
    parser.add_argument(
        "--judge-model",
        default="gpt-4o",# gpt-3.5-turbo
        help="Name of judge model.",
        choices=["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo", "text-davinci-003", "vicuna-api", "llama2-api", "mixtral-api", "phi2-api", "chatglm-api", "baichuan-api",
                 "gpt-3.5-turbo-instruct", "gpt-4o-mini", "gpt-4o", "no-judge"
                 ]
    )
    parser.add_argument(
        "--judge-max-n-tokens",
        type=int,
        default=30,
        help="Maximum number of tokens for the judge."
    )
    parser.add_argument(
        "--judge-temperature",
        type=float,
        default=0,
        help="Temperature to use for judge."
    )

    parser.add_argument("--baseline_method_email_complete_dataset_path", type=str, default="", help="baseline_method_email_complete_dataset_path")
    args = parser.parse_args()

    baseline_method_email_complete_dataset_path = args.baseline_method_email_complete_dataset_path
    baseline_method_email_complete_dataset= datasets.load_from_disk(baseline_method_email_complete_dataset_path)
    print("baseline_method_email_complete_dataset")
    print(baseline_method_email_complete_dataset)

    # Initialize judge
    judgeLM = GPTJudge(args)

    our_score_list = []
    baseline_score_list = []

    for index in range(len(baseline_method_email_complete_dataset)):
        baseline_truncate_gpt_email = baseline_method_email_complete_dataset[index]["truncate_gpt_email"]
        baseline_remain_gpt_email = baseline_method_email_complete_dataset[index]["remain_gpt_email"]
        baseline_score, baseline_raw_outputs = judgeLM.base_score(baseline_truncate_gpt_email, baseline_remain_gpt_email)
        baseline_score_list.append(baseline_score)
        print("baseline_remain_gpt_email-------------------------------")
        print(baseline_remain_gpt_email)
        print("baseline_raw_outputs---------------------------------------------------")
        print(baseline_raw_outputs)
        print("baseline_score:", baseline_score)

    mean_baseline_score = mean(baseline_score_list)
    print("len(baseline_score_list):", len(baseline_score_list))
    print("mean_baseline_score:", mean_baseline_score)        
