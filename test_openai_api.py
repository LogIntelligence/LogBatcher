from openai import OpenAI
import httpx


api_key = ""
client = OpenAI(
    api_key=api_key,
    http_client=httpx.Client(
        proxies="http://127.0.0.1:7890"
    ),
)

messages = []
messages.append(
    {"role": "user", "content": "say hello to the world"})
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=messages,
)

print(response)
