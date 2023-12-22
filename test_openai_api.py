from openai import OpenAI
import httpx
import re


def convert_to_template(code):
    # 提取引号中的内容
    pattern = r'\"(.+?)\"'
    match = re.search(pattern, code)
    if match:
        log_msg = match.group(1)
        # 将所有类似%d的占位符转化为<*>
        template = re.sub(r'%\w', '<*>', log_msg)
        return template
    else:
        return None


code = 'log_printf("Warning: Unrecognized packageExtended attribute.")'
template = convert_to_template(code)
print(template)


# api_key = "sk-MWCZbiYqiQUjacuGF53a6c71E3134177A585CeFe79D10aD2"
# client = OpenAI(
#     base_url="https://oneapi.xty.app/v1",
#     api_key=api_key,
#     http_client=httpx.Client(
#         proxies="http://127.0.0.1:7890"
#     ),
# )

# messages = []
# messages.append(
#     {"role": "user", "content": "say hello to the world"})
# response = client.chat.completions.create(
#     model="gpt-3.5-turbo",
#     messages=messages,
# )

# print(response)
