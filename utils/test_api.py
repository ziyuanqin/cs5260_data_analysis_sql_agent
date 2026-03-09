"""本地接口连通性测试脚本。

该文件有两个用途：
1) 手动测试模型流式输出
2) 被 app/config.py 解析为备用密钥来源
"""

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

# 使用兼容 OpenAI 协议的云雾接口创建客户端。
client = OpenAI(
    base_url="https://yunwu.ai/v1",
    api_key=key
)

# 发起流式生成请求，验证分片输出是否正常。
stream = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "user", "content": "你好？帮我生成一篇1000字的csgo新闻"},
    ],
    stream=True,
    timeout=100,
)

# 按到达顺序打印分片内容，模拟前端实时渲染效果。
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