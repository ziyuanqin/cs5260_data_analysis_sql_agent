# key='sk-8c5c5QqJno4M5valWVa3VzxRMLpDhI0FPjfCEU9pcYUPPlY8'


# from openai import OpenAI

# client = OpenAI(
#     base_url="https://yunwu.ai/v1",
#     api_key=key
# )

# response = client.chat.completions.create(
#   model="gpt-4o",
#   messages=[
#     {"role": "user", "content": "你好?"},

#   ],
#   timeout=100,

# )
# print(response)


key='sk-8c5c5QqJno4M5valWVa3VzxRMLpDhI0FPjfCEU9pcYUPPlY8'

from openai import OpenAI

client = OpenAI(
    base_url="https://yunwu.ai/v1",
    api_key=key
)

stream = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "user", "content": "你好？帮我生成一篇1000字的csgo新闻"},
    ],
    stream=True,
    timeout=100,
)

for chunk in stream:
    if not hasattr(chunk, "choices") or not chunk.choices:
        continue

    choice = chunk.choices[0]
    if not hasattr(choice, "delta") or choice.delta is None:
        continue

    content = getattr(choice.delta, "content", None)
    if content:
        print(content, end="", flush=True)

print()