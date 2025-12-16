# script count the tokens in a sentence 
import pandas as pd 
import tokenizers as tokenizer 
    
# themes data 
male_terms_data = '/Users/elda/Library/CloudStorage/OneDrive-ITU/Documents/gender_bias/evaluate-llm-gender-bias-ltc/evaluate_themes/csv_originals/male_to_female_clean_term_counts.csv'
female_terms_data = '/Users/elda/Library/CloudStorage/OneDrive-ITU/Documents/gender_bias/evaluate-llm-gender-bias-ltc/evaluate_themes/csv_originals/female_to_male_clean_term_counts.csv'

# siebert model data 
male_data = '/Users/elda/Library/CloudStorage/OneDrive-ITU/Documents/gender_bias/evaluate-llm-gender-bias-ltc/evaluate_sentiment/output_originals/siebert/male_to_female_clean_male.csv'
female_data = '/Users/elda/Library/CloudStorage/OneDrive-ITU/Documents/gender_bias/evaluate-llm-gender-bias-ltc/evaluate_sentiment/output_originals/siebert/female_to_male_clean_female.csv'

# regard model data 
regard_male = '/Users/elda/Library/CloudStorage/OneDrive-ITU/Documents/gender_bias/evaluate-llm-gender-bias-ltc/evaluate_sentiment/output_originals/regard/male_to_female_clean_male.csv'
regard_female = '/Users/elda/Library/CloudStorage/OneDrive-ITU/Documents/gender_bias/evaluate-llm-gender-bias-ltc/evaluate_sentiment/output_originals/regard/female_to_male_clean_female.'

# distilbert model data 
distilbert_male = '/Users/elda/Library/CloudStorage/OneDrive-ITU/Documents/gender_bias/evaluate-llm-gender-bias-ltc/evaluate_sentiment/output_originals/distilbert/male_to_female_clean_male.csv'
distilbert_female = '/Users/elda/Library/CloudStorage/OneDrive-ITU/Documents/gender_bias/evaluate-llm-gender-bias-ltc/evaluate_sentiment/output_originals/distilbert/female_to_male_clean_female.csv'

distilbert_data = pd.read_csv(distilbert_male)
print(distilbert_data.head)

# load female records
female_df = pd.read_csv(female_data)

# count the number of words in each sentence 
text = female_df.iloc[:, 0]
f_count_words = []
for line in text: 
    words = line.split()
    word_count = len(words)
    f_count_words.append(word_count)

# add columns to the data 
doc_begin = 50
female_df["sex"] = "f"
female_df["model"] = "siebert"
female_df["num_words"] = f_count_words
columns = ["doc_num", "text", "model", "pred", "label", "NEGATIVE", "POSITIVE", "sex", "num_words"]
female_df = female_df[columns]

# select 50 female 
female_df = female_df[(female_df["doc_num"] >= 0) & (female_df["doc_num"] < doc_begin)]

# load health terms type
terms_df = pd.read_csv(female_terms_data)
terms_df = terms_df[["doc_num", "female_count", "term_type"]]

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
        .query("female_count != 0")
        .groupby(["doc_num", "term_type"], as_index=False)
        .female_count.sum()
)

# find the dominant health term type
max_doc_count = terms_df.groupby("doc_num", as_index=False)["female_count"].max().rename(columns={"female_count": "max_count"})
terms_df = (
    terms_df
        .merge(max_doc_count, on="doc_num")
        .query("female_count == max_count")
        .drop(columns=["female_count"])
)

# merge female records and health term types
female_df = female_df.merge(
    terms_df[["doc_num", "term_type"]],  
    on="doc_num",
    how="left"
).rename(columns={"term_type": "health_term"})

# make csv female records file 
female_df.to_csv("female_records.csv", index=False)