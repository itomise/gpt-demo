from dotenv import load_dotenv
import openai
import re
import requests
import sys
from num2words import num2words
import os
import pandas as pd
import numpy as np
from openai.embeddings_utils import get_embedding, cosine_similarity
from transformers import GPT2TokenizerFast

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY") 
RESOURCE_ENDPOINT = os.getenv("OPENAI_API_BASE")

openai.api_type = "azure"
openai.api_key = API_KEY
openai.api_base = RESOURCE_ENDPOINT
openai.api_version = "2022-12-01"

url = openai.api_base + "/openai/deployments?api-version=2022-12-01"


# print(r.text)

# 2. load csv
df = pd.read_csv("./data/bill_sum_data.csv")
df_bills = df[['text', 'summary', 'title']]
df_bills

# 3. data cleaning
# s is input text
def normalize_text(s, sep_token = " \n "):
    s = re.sub(r'\s+',  ' ', s).strip()
    s = re.sub(r". ,","",s)
    # remove all instances of multiple spaces
    s = s.replace("..",".")
    s = s.replace(". .",".")
    s = s.replace("\n", "")
    s = s.strip()
    
    return s

df_bills['text'] = df_bills["text"].apply(lambda x : normalize_text(x))

# 長すぎる法案を削除
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
df_bills['n_tokens'] = df_bills["text"].apply(lambda x: len(tokenizer.encode(x)))
df_bills = df_bills[df_bills.n_tokens<2000]
# print(len(df_bills))

# トークン化を理解
understand_tokenization = tokenizer.tokenize(df_bills.text[0])
# print(understand_tokenization) 

# embedding
df_bills['curie_search'] = df_bills["text"].apply(lambda x : get_embedding(x, engine = 'text-search-curie-doc-001'))

# print(df_bills)

# search through the reviews for a specific product
def search_docs(df, user_query, top_n=3, to_print=True):
    embedding = get_embedding(
        user_query,
        engine="text-search-curie-query-001"
    )
    df["similarities"] = df.curie_search.apply(lambda x: cosine_similarity(x, embedding))

    res = (
        df.sort_values("similarities", ascending=False)
        .head(top_n)
    )
    if to_print:
        print(res)
    return res

res = search_docs(df_bills, "can i get information on cable company tax revenue", top_n=4)

