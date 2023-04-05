from dotenv import load_dotenv
load_dotenv()

import os
import requests
import json
import openai

openai.api_key = os.getenv('OPENAI_API_KEY')
openai.api_base = os.getenv('OPENAI_API_BASE')
openai.api_type = 'azure'
openai.api_version = '2022-12-01' # this may change in the future

deployment_name='gpt-35-test' #This will correspond to the custom name you chose for your deployment when you deployed a model. 


# tokenizer = openai.Tokenizer.create(engine=deployment_name)
# Send a completion call to generate an answer
print('Sending a test completion job')
start_phrase = '日本の首都はどこですか？'
response = openai.Completion.create(engine=deployment_name, prompt=start_phrase, max_tokens=10)
print(response)
text = response['choices'][0]['text'].replace('\n', '').replace(' .', '.').strip()
print(start_phrase+text)