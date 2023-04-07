from dotenv import load_dotenv
import openai
from num2words import num2words
import os
import numpy as np
import pandas as pd
from openai.embeddings_utils import get_embedding, cosine_similarity
from transformers import GPT2TokenizerFast
import time
import math

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
RESOURCE_ENDPOINT = os.getenv("OPENAI_API_BASE")

openai.api_type = "azure"
openai.api_key = API_KEY
openai.api_base = RESOURCE_ENDPOINT
openai.api_version = "2022-12-01"

target_texts = pd.read_csv('./_data/texts.csv')['text']

# demo
doc_model = "text-search-davinci-doc-001"
query_model = "text-search-davinci-query-001"

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
duration_sum = 0.0

def createEmbeddingInfo(text: str):
    start_time = time.time()
    get_embedding(text=text, engine=doc_model)
    duration = time.time() - start_time
    token = tokenizer.tokenize(text)
    global duration_sum
    duration_sum += duration
    return { "text_sub": text[:20], "text_length": len(text), "duration": duration, "token_length": len(token) }

embedded_target_texts = list(map(createEmbeddingInfo, target_texts))

for x in sorted(embedded_target_texts, key=lambda x: x['duration'], reverse=True):
    print(x)
    
print("duration 平均:", duration_sum / len(target_texts))
print("duration 中央値:", embedded_target_texts[math.floor(len(embedded_target_texts)/2)]['duration'])