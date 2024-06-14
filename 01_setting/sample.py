import json
from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {
            "role": "user",
            "content": "そばの原材料を教えて"
        }
    ],
    max_tokens=100,
    temperature=1,
    n=2
)

print(response)
# print(json.dumps(response, indent=2, ensure_ascii=False))