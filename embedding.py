from dotenv import load_dotenv
import os
import openai
import os
import numpy as np
import faiss
from transformers import AutoTokenizer, AutoModel
import requests

load_dotenv()
openai.api_type = "azure"
openai.api_key = os.getenv('OPENAI_API_KEY')
openai.api_base = os.getenv('OPENAI_API_BASE')
openai.api_version = "2022-12-01"

def create_embedding(text):
    response = openai.Embedding.create(
        input=text,
        engine="text-embedding-ada-002"
    )
    print(response)
    return response['data'][0]['embedding']

def main():
    documents = [
        "これは最初の文章です。",
        "これは二番目の文章です。",
        "これは三番目の文章です。",
        "これは最後の文章です。"
    ]
    data = np.vstack([np.array(create_embedding(doc)) for doc in documents]) 
    index = faiss.IndexFlatL2(data.shape[1])
    index.add(data)

    search_query = "これは三番目の文章です。"
    query_embedding = np.array(create_embedding(search_query)) 
    k = 3
    D, I = index.search(query_embedding[np.newaxis, :], k) 
    print("類似アイテムのインデックス:", I)
    print("類似アイテムとの距離:", D)
    # 類似アイテムのインデックスから元の文章を出力
    for idx, (sim_index, sim_distance) in enumerate(zip(I[0], D[0])):
        print(f"{idx + 1}. 類似度: {sim_distance:.4f}, 文章: {documents[sim_index]}")

main()
