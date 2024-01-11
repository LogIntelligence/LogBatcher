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

instruction_noindex = '''You will be provided with some log messages. You should check if the giving log messages share the same template. If so, abstract variables with ‘{placeholders}’ and return the template without additional explatation, otherwise return the templates'''

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



def get_responce(f, indexs, label, logs_temp):
    length = len(indexs)
    if length <= 5:
        f.write(f"---------------------------\n")
        f.write(f"cluster {label}: len={length}\n")
        f.write(f"---------------------------\n")
        pass
    else:
        for batch_logs in logs_temp:
            pass    
    template = {"template": "", "index":[]}
    print(f"cluster {label}: len={len(indexs)}")
    if label == 0 :
        for log in logs_temp[:20]:
            print(log)

# 读取CSV文件
dataset = 'Proxifier'
df = pd.read_csv(
    f'dataset/{dataset}/{dataset}_2k.log_structured_corrected.csv')
logs = df['Content'].tolist()
templates = [None for _ in range(2000)]

tokenized_logs = [tokenize(log) for log in logs]

k = 8

logs_label = []

for i in range(k):
    logs_label.append([])

labels = cluster(vectorize(tokenized_logs), k)

for i, label in enumerate(labels):
    logs_label[label].append(i)

f = open(f'test_Spark.txt', 'w')

for label, indexs in enumerate(logs_label):
    logs_temp = [logs[i] for i in indexs]
    get_responce(f, indexs, label, logs_temp)
