import openai
import os
import numpy as np
import faiss
from transformers import AutoTokenizer, AutoModel
import requests

openai.api_type = "azure"
openai.api_key = os.getenv('OPEN_AI_API_KEY')
openai.api_base = os.getenv('OPEN_AI_API_BASE')
openai.api_version = "2022-12-01"

# Sentence-BERTモデルの準備
model_name = "sentence-transformers/paraphrase-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def create_embedding(text: str):
    response = openai.Embedding.create(
        input=text,
        engine="text-embedding-ada-002"
    )
    return response['data'][0]['embedding']

def main():
    documents = [
        "これは最初の文章です。",
        "これは二番目の文章です。",
        "これは最後の文章です。"
    ]
    data = np.vstack([np.array(create_embedding(doc)) for doc in documents]) 
    index = faiss.IndexFlatL2(data.shape[1])
    index.add(data)

    search_query = "これは関連する文章です。"
    query_embedding = np.array(create_embedding(search_query)) 
    k = 3
    D, I = index.search(query_embedding[np.newaxis, :], k) 
    print("類似アイテムのインデックス:", I)
    print("類似アイテムとの距離:", D)

main()
