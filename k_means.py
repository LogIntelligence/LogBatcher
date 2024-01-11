import pandas as pd
import re
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer


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


def get_responce(indexs, label, logs_temp):
    length = len(indexs)
    # if length <= 20:
    #     # strightly parsing
    #     pass
    # else:
    #     # batching parsing
    #     pass
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

for label, indexs in enumerate(logs_label):
    logs_temp = [logs[i] for i in indexs]
    get_responce(indexs, label, logs_temp)
