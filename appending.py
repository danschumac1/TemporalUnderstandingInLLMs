import pandas as pd
import json
import numpy as np
import openai
from openai import OpenAI
from dotenv import load_dotenv
import os
import json
from functions.homebrew import extract_output_values_from_json_file
# os.chdir('./DeepLearning/Temporal')

rel_output = extract_output_values_from_json_file('./data/output/rel_context.out')

TQ = pd.read_csv('./data/TQ.csv', encoding='ISO-8859-1')
TQ['PromptHead'] = [f'Question: {q}\n\nAnswer: {a}\n\Generated Context: ' for q, a in zip(TQ['Question'],TQ['Answer'])]
# TQ.loc(:'rel_C') = rel_output
TQ['relevant_context'] = rel_output

# =============================================================================
# Spacy # IDENTIFY DATES AND SPLIT THEM INTO UNIQUE ELEMENTS IN LIST
# =============================================================================
import spacy

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

list_of_dates = []
for relevant_context in TQ['relevant_context']:
    doc = nlp(relevant_context)
    dates = [ent.text for ent in doc.ents if ent.label_ == "DATE"]
    list_of_dates.append(dates)

# generate a random year between 1850 and 2023
# Define the range of years
def generate_false_year(actual_year):
    start_year = 1850
    end_year = 2023

    # Generate a list of years excluding the specified year
    years = [year for year in range(start_year, end_year + 1) if year != actual_year]

    # Randomly select a year from the list
    false_year = np.random.choice(years)
    return false_year

# =============================================================================
# MAKE USEFUL FUNCTIONS
# =============================================================================

# generate a random month Janurary - Feburary
def generate_false_month(actual_month):
    months = [
        "January", "February", "March",
        "April", "May", "June",
        "July", "August","September",
        "October", "November", "December"
    ]

    filtered_months = [month for month in months if month != actual_month]

    false_month = np.random.choice(filtered_months)
    return false_month

months = [
    "January", "February", "March",
    "April", "May", "June",
    "July", "August","September",
    "October", "November", "December"
]


# =============================================================================
# GENERATE FALSE DATES
# =============================================================================

import re
year_pattern = r'\b\d{4}\b'

fab_dates_for_whole_data = []

for lst in list_of_dates:
    # print(f'Processing list:', lst, '\n')
    fab_list_of_dates_for_post = []
    for string in lst:
        new_string = ''
        # print('Processing string:', string, '\n')
        for sub_str in string.split():
            # print('Processing item:', sub_str, '\n')
            if sub_str in months:
                # print('It is a month:', sub_str, '\n')
                false_month = generate_false_month(sub_str)
                new_string += false_month + " "
            elif re.match(year_pattern, sub_str):
                # print('It is a year!!!', sub_str)
                false_year = str(generate_false_year(int(sub_str)))
                new_string += false_year + " "
            else:
                new_string += sub_str + " "
        fab_list_of_dates_for_post.append(new_string.strip())

    fab_dates_for_whole_data.append(fab_list_of_dates_for_post)
    # print('Modified list:', fab_list_of_dates_for_post)

    # break  # Remove break if you want to process all lists in list_of_dates


# # Double check work 
# for t,f in zip(list_of_dates, fab_dates_for_whole_data):
#     print( 'ACTUAL:', t, '\n','FABRICATED:', f )
#     print('\n')

# =============================================================================
# MAP FALSE DATES BACK TO CONTEXTS
# =============================================================================

# Initialize the list to store contexts with dates replaced
wrong_date_context = []

# Iterate over each context
for context in TQ['relevant_context']:
    modified_context = context  # Start with the original context
    # Iterate over each set of actual and false dates
    for actual_date_list, false_date_list in zip(list_of_dates, fab_dates_for_whole_data):
        # Replace each actual date with its corresponding false date
        for actual_date, false_date in zip(actual_date_list, false_date_list):
            modified_context = re.sub(re.escape(actual_date), false_date, modified_context)
    # Append the modified context to the list
    wrong_date_context.append(modified_context)
 

# At this point, wrong_date_context contains all the contexts with dates replaced

# Double check work 
for t,f in zip(TQ['relevant_context'], wrong_date_context):
    print( 'ACTUAL:', t, '\n\n','FABRICATED:', f )
    print('\n')
    break

TQ['wrong_date_context'] = wrong_date_context
TQ = TQ.rename(columns={'PromptHead':'QA'})

TQ.to_csv('./data/TQ_revised.csv',  index = False)
