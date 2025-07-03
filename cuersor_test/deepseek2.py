# Please install OpenAI SDK first: `pip3 install openai`
from openai import OpenAI

API_KEY = "sk-391b572da60c40bea40fa8f75f650fd8"
#client = OpenAI(api_key="<DeepSeek API Key>", base_url="https://api.deepseek.com")
client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
    ],
    stream=False
)

print(response.choices[0].message.content)