"""
Created on 04/08/2024

@author: Dan Schumacher
"""
import os
os.getcwd()
os.chdir('./TemporalUnderstandingInLLMs')
#endregion
#region # IMPORTS
# =============================================================================
# IMPORTS
# =============================================================================
import pandas as pd
from functions.homebrew import extract_output_values_from_json_file
from transformers import AutoTokenizer
import json
import wandb


#endregion
#region # DATA READING
# =============================================================================
# DATA READING
# =============================================================================
dataset_file_path = './data/TQ_revised.csv'
df = pd.read_csv(dataset_file_path)
df.to_json('./data/TQ_revised.json')

with open('./data/TQ_revised.json', 'r') as f:
    data_dict = [json.load(f)];



# LOG TO W&B
with wandb.init(project="alpaca_ft"):
    at = wandb.Artifact(
        name="TemporalUnderstandingInLLMs", 
        type="dataset",
        description="Temporal Questions Dataset",
        metadata={"Original Source":"@$@ put URL HERE"},
    )
    at.add_file("./data/TQ_revised.json")
    
    table = wandb.Table(columns=list(data_dict[0].keys()))
    for row in data_dict:
        table.add_data(*row.values())

#endregion
#region # DATA PREPROCESSING
# =============================================================================
# DATA PREPROCESSING
# =============================================================================
# NO CONTEXT
def create_no_context_prompt(row):
    return (
        "Below is a Question that needs to be answered. "
        "Write a response that answers the question to the best of your abilities.\n\n"
        "### Question:\n{Question}\n\n### Response:\n"
            ).format_map(row)

no_context_prompt = [create_no_context_prompt(row) for row in data_dict]  

# RELEVANT CONTEXT
def create_rel_context_prompt(row):
    return (
        "Below is a Question, paired with a input that provides further context."
        "Write a response that answers the question given the response to the best of your abilities.\n\n"
        "### Question:\n{Question}\n\n### Input:\n{relevant_context}\n\n### Response:\n"
        ).format_map(row)

rel_context_prompt = [create_rel_context_prompt(row) for row in data_dict]


# WRONG DATE CONTEXT
def create_wrongDate_context_prompt(row):
    return (
        "Below is a Question, paired with a input that provides further context."
        "Write a response that answers the question given the response to the best of your abilities.\n\n"
        "### Question:\n{Question}\n\n### Input:\n{wrong_date_context}\n\n### Response:\n"
        ).format_map(row)

wrongDate_context_prompt = [create_wrongDate_context_prompt(row) for row in data_dict]

#region #  @$@ COMMENTED OUT FOR LATER


# # RANDOM CONTEXT
# def create_random_context_prompt(row):
#     return (
#         "Below is a Question, paired with a input that provides further context."
#         "Write a response that answers the question given the response to the best of your abilities.\n\n"
#         "### Question:\n{Question}\n\n### Input:\n{relevant_context}\n\n### Response:\n"
#         ).format_map(row)

# # WRONG DATA RANDOM CONTEXT
# # RANDOM CONTEXT
# def create_wrongDateRandom_context_prompt(row):
#     return (
#         "Below is a Question, paired with a input that provides further context."
#         "Write a response that answers the question given the response to the best of your abilities.\n\n"
#         "### Question:\n{Question}\n\n### Input:\n{relevant_context}\n\n### Response:\n"
#         ).format_map(row)
#endregion
#endregion

