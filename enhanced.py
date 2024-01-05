from openai import OpenAI
from tqdm import tqdm
import httpx
import pandas as pd
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
import openai
import backoff

api_key = "sk-MWCZbiYqiQUjacuGF53a6c71E3134177A585CeFe79D10aD2"
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

instruction = '''Given the following raw log, you should generate the corresponding logging statement. The logging statement is a single line of code that is used to log a message. You need to understand the information in the raw log, determine the log level, extract all possible variables and replace them with placeholders, and finally generate the logging statement using the correct logging function and the log message with placeholders. All extracted variables should be passed as arguments to the logging function.'''

demonstrations = {
    'Spark': [{"role": "user", "content": 'Log: 17/06/09 20:10:48 INFO storage.MemoryStore: Block broadcast_0 stored as values in memory (estimated size 384.0 B, free 317.5 KB)'}, {"role": "assistant", "content": 'Logging statement: logger.info("Block {} stored as values in memory (estimated size {}, free {})", "broadcast_0", "384.0 B", "317.5 KB")'}],
    'Hadoop': [{"role": "user", "content": 'Log: 2015-10-18 18:01:53,885 INFO [AsyncDispatcher event handler] org.apache.hadoop.mapreduce.v2.app.job.impl.TaskAttemptImpl: attempt_1445144423722_0020_m_000000_0 TaskAttempt Transitioned from NEW to UNASSIGNED'}, {"role": "assistant", "content": 'Logging statement: logger.info("{} TaskAttempt Transitioned from NEW to UNASSIGNED", "attempt_1445144423722_0020_m_000000_0")'}],
    'Linux': [{"role": "user", "content": 'Log: Jun 15 20:05:31 combo sshd(pam_unix)[24141]: authentication failure; logname= uid=0 euid=0 tty=NODEVssh ruser= rhost=d211-116-254-214.rev.krline.net'}, {"role": "assistant", "content": 'Logging statement: logger.info("authentication failure; logname= uid={} euid={} tty={} ruser= rhost={}", "0", "0", "NODEVssh", "d211-116-254-214.rev.krline.net")'}],
}

messages = [{"role": "system", "content": instruction}]

df = pd.DataFrame()

datasets_now = ['Linux']

# 打开文件
f = open('temp.txt', 'w')

dataset = 'Hadoop'
# for dataset in datasets_now:

with open(f'dataset/{dataset}/{dataset}_2k.log', 'r') as logfile:
    logs = logfile.readlines()
df['Log'] = logs

LoggingStatements = []
messages.append(demonstrations[dataset][0])
messages.append(demonstrations[dataset][1])

print('------------------------------------')
print('dataset: ', dataset)
print('------------------------------------')

for log in tqdm(logs):
    messages.append({"role": "user", "content": log})
    response = get_responce(messages)
    LoggingStatements.append(response)
    f.write("%s\n" % response)
    messages.pop()

df['EventTemplate'] = pd.read_csv(f'dataset/{dataset}/{dataset}_2k.log_structured_corrected.csv')['EventTemplate']
df['LoggingStatement'] = LoggingStatements
df.to_csv(f'outputs/enhanced_gpt/1shot/{dataset}.csv', index=False)
f = open('temp.txt', 'w')
f.close()
messages.pop()
messages.pop()  

