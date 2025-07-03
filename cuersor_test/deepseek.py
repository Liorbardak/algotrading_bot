# deepseek.py

import requests

# Enter your API Key
API_KEY = "sk-391b572da60c40bea40fa8f75f650fd8"

url = "https://api.deepseek.com/chat/completions"
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}

data = {
    "model": "deepseek-reasoner",  # Use 'deepseek-reasoner' for R1 model or 'deepseek-chat' for V3 model
    "messages": [
        {"role": "system", "content": "You are a professional assistant"},
        {"role": "user", "content": "Who are you?"}
    ],
    "stream": False  # Disable streaming
}

response = requests.post(url, headers=headers, json=data)

if response.status_code == 200:
    result = response.json()
    print(result['choices'][0]['message']['content'])
else:
    print("Request failed, error code:", response.status_code)