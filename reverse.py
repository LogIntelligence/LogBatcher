import httpx
from openai import OpenAI
from tqdm import tqdm
from config import extract_and_replace
import pandas as pd
import openai
import backoff
import json

api_key = "sk-ShmyeH9VjAnRuT1S55A71a9fC69640948d20F73bA634C3A5"
client = OpenAI(
    base_url="https://oneapi.xty.app/v1",  # 中转url
    api_key=api_key,                      # api_key
    http_client=httpx.Client(
        proxies="http://127.0.0.1:7890"  # 代理地址
    ),
)

@backoff.on_exception(backoff.expo, (openai.APIStatusError, openai.InternalServerError), max_tries=5)
def get_responce(messages):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
    )
    return response.choices[0].message.content.strip('\n')


instruction = '''Given the following raw log, you should generate the corresponding logging statement. The logging statement is a single line of code that is used to log a message. You need to understand the information in the raw log, extract all possible variables and replace them with placeholders, and finally generate the logging statement. All extracted variables should be passed as arguments to the logging function.
Print the input log's template delimited by backticks.'''

datasets = ['BGL', 'HDFS', 'Linux', 'HealthApp', 'OpenStack', 'OpenSSH', 'Proxifier', 'HPC', 'Zookeeper', 'Mac',
            'Hadoop', 'Android', 'Windows', 'Apache', 'Thunderbird', 'Spark']
datasets = ['BGL']

# load demonstrations
with open('demonstrations.json', 'r') as f:
    demonstrations = json.load(f)

for dataset in datasets:
    messages = [{"role": "system", "content": instruction}]
    df =  pd.read_csv(f'dataset/{dataset}/{dataset}_2k.log_structured_corrected.csv')
    logs = df['Content']

    # demonstrations added
    messages.append(demonstrations[dataset][0])
    messages.append(demonstrations[dataset][1])

    # show dataset name
    print('-' * 20)
    print(f"dataset: {dataset}")
    print('-' * 20)

    outputs = []
    for log in tqdm(logs):
        messages.append({"role": "user", "content": 'Log: ' + log})
        response = get_responce(messages)
        # process response
        output = extract_and_replace(response)
        outputs.append(output)
        messages.pop()
    # write to file
    if(len(outputs) != len(logs)):
        print('error')
        with open(f'outputs/enhanced_gpt/1shot/{dataset}_error.txt', 'a') as f:
            for output in outputs:
                f.write(output + '\n')
        exit()
    else:
        df['Output'] = outputs
        df[['Content', 'EventTemplate', 'Output']].to_csv(f'outputs/enhanced_gpt/1shot/{dataset}.csv', index=False)

