# script count the tokens in a sentence 
import pandas as pd 
import tokenizers as tokenizer

# Data sets
male_data = '/Users/elda/Library/CloudStorage/OneDrive-ITU/Documents/gender_bias/evaluate-llm-gender-bias-ltc/evaluate_sentiment/output_originals/siebert/male_to_female_clean_male.csv'
female_data = '/Users/elda/Library/CloudStorage/OneDrive-ITU/Documents/gender_bias/evaluate-llm-gender-bias-ltc/evaluate_sentiment/output_originals/siebert/female_to_male_clean_female.csv'
male_terms_data = '/Users/elda/Library/CloudStorage/OneDrive-ITU/Documents/gender_bias/evaluate-llm-gender-bias-ltc/evaluate_themes/csv_originals/male_to_female_clean_term_counts.csv'
female_terms_data = '/Users/elda/Library/CloudStorage/OneDrive-ITU/Documents/gender_bias/evaluate-llm-gender-bias-ltc/evaluate_themes/csv_originals/female_to_male_clean_term_counts.csv'

# load female records
male_df = pd.read_csv(male_data)
print(male_df.head)

# count the number of words in each sentence 
text = male_df.iloc[:, 0]
m_count_words = []
for line in text: 
    words = line.split()
    word_count = len(words)
    m_count_words.append(word_count)

# add columns to the data 
doc_size = 50
male_df["sex"] = "m"
male_df["model"] = "siebert"
male_df["num_words"] = m_count_words
male_df["doc_num"] = male_df["doc_num"] + doc_size
columns = ["doc_num", "text", "model", "pred", "label", "NEGATIVE", "POSITIVE", "sex", "num_words"]
male_df = male_df[columns]

# select 50 male 
male_df = male_df[(male_df["doc_num"] >= doc_size) & (male_df["doc_num"] < (doc_size*2))]

# load health terms type
terms_df = pd.read_csv(male_terms_data)
terms_df = terms_df[["doc_num", "male_count", "term_type"]]

""" 
Extract the counts of health term_type for each patient. 
There are 3 different types of health terms 
- physical_health 
- physical_appearance
- mental_health

A patient can have all 3 types of health terms but often physical health is dominant. 
when counting the number of sub terms. Examples of terms are 'blood preassure' or 'fracture'. 
These terms belong to the term type 'physical health'. 
"""
terms_df = (
    terms_df
        .query('term_type != "subjective_language" ')
        .query("male_count != 0")
        .groupby(["doc_num", "term_type"], as_index=False)
        .male_count.sum()
)

# find the dominant health term type
max_doc_count = terms_df.groupby("doc_num", as_index=False)["male_count"].max().rename(columns={"male_count": "max_count"})
terms_df = (
    terms_df
        .merge(max_doc_count, on="doc_num")
        .query("male_count == max_count")
        .drop(columns=["male_count"])
)

# merge female records and health term types
male_df = male_df.merge(
    terms_df[["doc_num", "term_type"]],  
    on="doc_num",
    how="left"
).rename(columns={"term_type": "health_term"})
print(male_df.head)

# make csv female records file 
male_df.to_csv("male_records.csv", index=False)