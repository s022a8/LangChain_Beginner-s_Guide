import json
from openai import OpenAI

client = OpenAI()

response = client.completions.create(
    model="gpt-3.5-turbo-instruct",
    prompt="今日の天気がとても良くて、気分が",
    stop="。",
    max_tokens=100,
    temperature=0.5,
    n=2
)

print(response)
# print(json.dumps(response, indent=2, ensure_ascii=False))