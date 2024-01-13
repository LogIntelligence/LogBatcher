import random
import pandas as pd
import re
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from openai import OpenAI
import openai
import backoff

api_key = "sk-ShmyeH9VjAnRuT1S55A71a9fC69640948d20F73bA634C3A5"
client = OpenAI(
    base_url="https://oneapi.xty.app/v1",  # 中转url
    api_key=api_key,                      # api_key
)

 # instruction_noindex = '''You will be provided with some log messages. You should check whether the giving log messages share the same template. If so, abstract variables with ‘{placeholders}’ and return the template without additional explatation, otherwise return 'false' only.'''

instruction_noindex = '''You will be provided with some log messages. You should check whether the giving log messages share the same template. If so, abstract variables with ‘{placeholders}’ and return the template without additional explatation'''

instruction2 = '''Giving some log tempaltes, the AI assistant should merge the possibly same templates'''

def tokenize(log_content):
    words = log_content.split()
    words = [word for word in words if not re.search(r'\d', word)]
    return words

def vectorize(tokenized_logs):
    vectorizer = TfidfVectorizer(tokenizer=lambda x: x, lowercase=False)
    return vectorizer.fit_transform(tokenized_logs)

def cluster(vectorized_logs, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(vectorized_logs)
    return kmeans.labels_


@backoff.on_exception(backoff.expo, (openai.APIStatusError, openai.InternalServerError), max_tries=5)
def chat(messages):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
    )
    return response.choices[0].message.content.strip('\n')

# 预处理
def preprocessing(answer):
    # 按照'\n'分割字符串成一个列表
    lines = answer.split('\n')
    # 检查列表的每一项，如果不含有'template'，则保留
    lines_without_template = [re.sub(r'\{.*?\}', '<*>', line) for line in lines if 'template' not in line]
    # 将处理后的列表重新组合成一个字符串
    return lines_without_template

def batch_parsing(batch_logs):
    templates = []
    messages = []
    messages.append({"role": "system", "content": instruction_noindex})
    # batch logs to str
    prompt = ""
    for log in batch_logs:
        prompt += log + '\n'
    messages.append({"role": "user", "content": prompt})
    answer = chat(messages)
    return preprocessing(answer)

def get_responce(f, indexs, label, logs_temp, k):
    length = len(indexs)
    templates = []
    random.shuffle(logs_temp)

    # get all templates
    if length <= 5:
        f.write(f"---------------------------\n")
        f.write(f"cluster {label}: len={length}\n")
        f.write(f"---------------------------\n")
        print(f"cluster {label}: len={length}")
        pass
    else:
        # 按20个一批分批处理
        for i in range(0, len(logs_temp), k):
            batch = logs_temp[i:i+k]
            templates_batch = batch_parsing(batch)
            for template in templates_batch:
                if template not in templates:
                    templates.append(template)

        f.write(f"---------------------------\n")
        f.write(f"cluster {label}: len={length}\n")
        print(f"cluster {label}: len={length}")
        for template in templates:
            f.write(f"{template}\n")
            print(template)
        f.write(f"---------------------------\n")
# 读取CSV文件
dataset = 'Hadoop'
df = pd.read_csv(
    f'dataset/{dataset}/{dataset}_2k.log_structured_corrected.csv')
logs = df['Content'].tolist()
templates = [None for _ in range(2000)]

tokenized_logs = [tokenize(log) for log in logs]

k = 115

logs_label = []

for i in range(k):
    logs_label.append([])

labels = cluster(vectorize(tokenized_logs), k)

for i, label in enumerate(labels):
    logs_label[label].append(i)

f = open(f'test_Spark.txt', 'w')

for label, indexs in enumerate(logs_label):
    logs_temp = [logs[i] for i in indexs]
    get_responce(f, indexs, label, logs_temp, 20)

f.close()
