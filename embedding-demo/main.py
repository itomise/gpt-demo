from dotenv import load_dotenv
import openai
from num2words import num2words
import os
import numpy as np
import pandas as pd
from openai.embeddings_utils import get_embedding, cosine_similarity
from transformers import GPT2TokenizerFast
import time

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
RESOURCE_ENDPOINT = os.getenv("OPENAI_API_BASE")

openai.api_type = "azure"
openai.api_key = API_KEY
openai.api_base = RESOURCE_ENDPOINT
openai.api_version = "2022-12-01"

target_texts = pd.read_csv('./_data/texts.csv')['text']

start_time = time.time()

# demo
doc_model = "text-search-ada-doc-001"
query_model = "text-search-ada-query-001"

embedded_target_texts = list(map(lambda x: get_embedding(x, engine=doc_model), target_texts))

question = "この契約書で甲はいくらのお金を払いますか？"
embedded_question = get_embedding(question, engine=query_model)

similarities_objects = []

for i, x in enumerate(embedded_target_texts):
    similarities_objects.append({ "origin_text": target_texts[i][:20], "similarities": cosine_similarity(x, embedded_question) })

print("かかった時間 (秒) :", time.time() - start_time)

print("question :", question[:20])
sorted_similarities_objects = sorted(similarities_objects, key=lambda x: x['similarities'], reverse=True)
for obj in sorted_similarities_objects:
    print(obj)