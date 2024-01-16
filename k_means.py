from collections import Counter
import os
import random
import time
import pandas as pd
import re
from sklearn.cluster import KMeans, DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from openai import OpenAI
import openai
import backoff
import httpx
from post_process import correct_single_template

api_key = "sk-KlLjOnG8myjg3Hhg5dF71699E6814e8b9753F00bB076C400"
client = OpenAI(
    base_url="https://oneapi.xty.app/v1",  # 中转url
    api_key=api_key,                      # api_key
    http_client=httpx.Client(
        proxies="http://127.0.0.1:7890"  # 代理地址
    ),
)

# instruction_noindex = '''You will be provided with some log messages. You should check whether the giving log messages share the same template. If so, abstract variables with ‘{placeholders}’ and return the template without additional explatation, otherwise return 'false' only.'''

instruction_noindex = '''You will be provided with some log messages. You should check whether the giving log messages share the same template. If so, abstract variables with `{placeholders}` and return the template without additional explatation'''

instruction_new = '''You will be provided with some log messages. You should check if the giving log messages share the same template. If so, abstract variables with `{{placeholders}}` to extract the corresponding template.
Print the input log's template delimited by backticks.
'''

instruction_for_one_log = '''You will be provided with a log message delimited by backticks. You must abstract variables with `{{placeholders}}` to extract the corresponding template.
Print the input log's template delimited by backticks.'''

instruction2 = '''Giving some log tempaltes, the AI assistant should merge the possibly same templates'''


def tokenize(log_content):
    words = re.split('[ ,]', log_content)
    # words = log_content.split()
    words = [word for word in words if not re.search(r'\d', word)]
    return words


def vectorize(tokenized_logs):
    vectorizer = TfidfVectorizer(tokenizer=lambda x: x, lowercase=False)
    return vectorizer.fit_transform(tokenized_logs)


def cluster(vectorized_logs, num_clusters = '10', method='kmeans'):
    if method == 'kmeans':
        kmeans = KMeans(n_clusters=num_clusters)
        kmeans.fit(vectorized_logs)
        return kmeans.labels_
    if method == 'dbscan':
        dbscan = DBSCAN(eps=0.3, min_samples=5)
        dbscan.fit(vectorized_logs)
        return dbscan.labels_


@backoff.on_exception(backoff.expo, (openai.APIStatusError, openai.InternalServerError), max_tries=5)
def chat(messages):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
    )
    return response.choices[0].message.content.strip('\n')

# 预处理


def postprocessing(response, isafter = False):
    boolean = {'true', 'false'}
    default_strings = {'null', 'root', 'admin'}
    path_delimiters = {  # reduced set of delimiters for tokenizing for checking the path-like strings
        r'\s', r'\,', r'\!', r'\;', r'\:',
        r'\=', r'\|', r'\"', r'\'',
        r'\[', r'\]', r'\(', r'\)', r'\{', r'\}'
    }
    token_delimiters = path_delimiters.union({  # all delimiters for tokenizing the remaining rules
        r'\.', r'\-', r'\+', r'\@', r'\#', r'\$', r'\%', r'\&',
    })

    response = response.strip().strip('\n')
    if "\n\n" in response:
        response = response.split("\n\n")[0]
    reg = re.compile("`([^`]+)`")
    tmps = reg.findall(response)
    tmps = [x.strip('\n').strip() for x in tmps]
    tmp = ''
    if len(tmps) == 1:
        tmp = tmps[0]
    if len(tmps) > 1:
        tmp = max(tmps, key=len)
    
    tmp = tmp.strip('\n').strip()
    tmp = re.sub(r'\{\{.*?\}\}', '<*>', tmp)
    template = tmp
    if not isafter:
        template = correct_single_template(template)
    if isafter:
        tokens = template.split(' ')
        for i in range(len(tokens)):
            if re.match(r'^\d+$', tokens[i]):
                tokens[i] = '<*>'
            for word in default_strings.union(boolean):
                tokens[i] = re.sub(r'(?i)(?<![a-z])' + word + r'(?![a-z])','<*>', tokens[i], flags=re.IGNORECASE)
            
            if tokens[i].count('<*>') >= 2:
                if tokens[i].startswith('/'):
                    tokens[i] = tokens[i][1:]
                # 保留前后的符号
                else:
                    prefix = '' if not re.match(
                        r'^[\[\]\.\:\,\/\']', tokens[i]) else tokens[i][0]
                    suffix = '' if not re.match(
                        r'.*[\[\]\.\:\,\/\']$', tokens[i]) else tokens[i][-1]
                    tokens[i] = prefix + '<*>' + suffix
        template = ' '.join(tokens)
    return template


def batch_parsing(batch_logs):

    # if all logs's length is 1, and not contain any digit, return the log itself
    if all(len(log) == 1 and not any(char.isdigit() for char in log[0]) for log in batch_logs):
        return batch_logs[0]

    messages = []
    messages.append({"role": "system", "content": instruction_new})
    # batch logs to str
    prompt = ""
    for log in batch_logs:
        prompt += log + '\n'
    messages.append({"role": "user", "content": prompt})
    answer = chat(messages)
    return postprocessing(answer , isafter=False)


def choose(templates):

    # 使用Counter计算频率
    freq = Counter(templates)
    length = len(freq) 
    candidates = freq.most_common(len(freq))
    final_template = ''
    if length == 0:
        pass
    elif length == 1 or candidates[0][1] > candidates[1][1]:
        final_template = candidates[0][0]
    else:
        count1 = 0
        count2 = 0
        for char in candidates[0][0]:
            if char.isdigit():
                count1 += 1
        for char in candidates[1][0]:
            if char.isdigit():
                count2 += 1
        if count1 < count2:
            final_template = candidates[1][0]
    return final_template, freq

def get_responce(f, indexs, label, logs_temp, k, ground_truth, israndom=True):
    length = len(indexs)
    templates = []
    if israndom:
        seed = time.time()
        random.seed(seed)
        random.shuffle(indexs)
        random.seed(seed)
        random.shuffle(logs_temp)

    for i in range(0, len(logs_temp), k):
        batch = logs_temp[i:i+k]
        template = batch_parsing(batch)
        if template != '':
            templates.append(template)

    f.write(f"---------------------------\n")
    f.write(f"cluster {label}: len={length}\n")
    f.write(f"{ground_truth} (ground truth)\n")
    print(f"cluster {label}: len={length}")

    output_tempalte, freq = choose(templates)
        
    # 打印结果
    for key, value in freq.items():
        f.write(f"{key}: {value}\n")
        print(f"{key}: {value}")
    f.write(f"---------------------------\n")
    return indexs, output_tempalte


def single_dataset_paring(dataset, output_dir, k = 10, cluster_method='kmeans'):
    os.makedirs(output_dir, exist_ok=True)
    # 读取日志文件
    df = pd.read_csv(
        f'dataset/{dataset}/{dataset}_2k.log_structured_corrected.csv')
    logs = df['Content'].tolist()

    # 对logs分词，得到logs_label，二维列表
    tokenized_logs = [tokenize(log) for log in logs]
    

    # 聚类
    labels = cluster(vectorize(tokenized_logs), k, cluster_method)
    cluster_nums = max(labels) + 1
    logs_label = []
    for i in range(cluster_nums):
        logs_label.append([])
    for i, label in enumerate(labels):
        if label != -1:
            logs_label[label].append(i)

    f = open(output_dir+ f'{dataset}.txt', 'w')
    outputs = [None for _ in range(2000)]
    for label, indexs in enumerate(logs_label):
        logs_temp = [logs[i] for i in indexs]
        ground_truth = df['EventTemplate'][indexs[0]]
        indexs_after, template = get_responce(
            f, indexs, label, logs_temp, 50, ground_truth)
        for index in indexs_after:
            outputs[index] = template

    f.close()
    df['Output'] = outputs
    df[['Content', 'EventTemplate', 'Output']].to_csv(output_dir+
        f'{dataset}.csv', index=False)


# main
if __name__ == "__main__":
    datasets = ['BGL', 'HDFS', 'Linux', 'HealthApp', 'OpenStack', 'OpenSSH', 'Proxifier', 'HPC', 'Zookeeper', 'Mac',
                'Hadoop', 'Android', 'Windows', 'Apache', 'Thunderbird', 'Spark']
    datasets = ['Linux', 'HealthApp', 'OpenStack', 'Proxifier', 'HPC', 'Zookeeper', 'Mac',
                'Hadoop', 'Android', 'Windows', 'Thunderbird', 'Spark']
    cluster_nums = [132, 14, 143, 71, 56, 180, 14, 51, 54, 350, 115, 189, 57, 6, 194, 38]
                 # [120, 14, 116, 75, 43,  26,  8, 46, 50, 341, 114, 158, 50, 6, 149, 36]
    output_dir = 'outputs/k_means/Second/'
    for index, dataset in enumerate(datasets[:2]):
        k = cluster_nums[index]
        single_dataset_paring(dataset, output_dir, cluster_method='dbscan',
                              )
    
    # single_dataset_paring('Linux', cluster_method='dbscan')
