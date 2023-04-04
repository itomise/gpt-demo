#Note: The openai-python library support for Azure OpenAI is in preview.
from dotenv import load_dotenv
import os
import openai

load_dotenv()

openai.api_type = "azure"
openai.api_key = os.getenv('OPEN_AI_API_KEY')
openai.api_base = os.getenv('OPEN_AI_API_BASE')
openai.api_version = "2023-03-15-preview"

response = openai.ChatCompletion.create(
  engine="gpt-35-test", # replace this value with the deployment name you chose when you deployed the associated model.
  messages = [
      [{"role":"system","content":"あなたはプロのエンジニアです。質問に対して簡潔に回答します。\nまた、足りない情報などがあればユーザーに対して質問も作成します。"},
       {"role":"user","content":"python のメリットデメリットをいくつか簡潔に教えてください。"},
       {"role":"assistant","content":"Pythonのメリット:\n- 読みやすいコードが書ける\n- シンプルな文法で書ける\n- 大量のライブラリがある\n- マルチプラットフォームで動作する\n- データ解析や機械学習に向いている\n\nPythonのデメリット:\n- 実行速度が遅い\n- 型の宣言が不要であるため、プログラムのエラーが実行時に発生する可能性がある\n- コードのインデントが重要であるため、初心者にとっては学習コストが高い場合がある"}]
  ]
  ,
  temperature=0,
  max_tokens=350,
  top_p=0.95,
  frequency_penalty=0,
  presence_penalty=0,
  stop=None)

print(response)