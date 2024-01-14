from collections import Counter
import random
import time
import pandas as pd
import re
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from openai import OpenAI
import openai
import backoff
import httpx

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

instruction_for_one_log = '''You will be provided with a log message delimited by backticks. You must abstract variables with `{{placeholders}}` to extract the corresponding template.
Print the input log's template delimited by backticks.'''

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


def postprocessing(response):
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

    template = response.strip().strip('\n')
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

    tokens = tmp.split(' ')
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
    return ' '.join(tokens)


def batch_parsing(batch_logs):
    messages = []
    messages.append({"role": "system", "content": instruction_noindex})
    # batch logs to str
    prompt = ""
    for log in batch_logs:
        prompt += log + '\n'
    messages.append({"role": "user", "content": prompt})
    answer = chat(messages)
    return preprocessing(answer)

def choose(candidates):
    words1 = candidates[0][0].split(' ')
    words2 = candidates[1][0].split(' ')
    count1 = 0
    count2 = 0
    for element in words1:
        if any(char.isdigit() for char in element):
            count1 += 1
    for element in words2:
        if any(char.isdigit() for char in element):
            count2 += 1
    if count1 > count2:
        return candidates[1][0]
    else:
        return candidates[0][0]

def get_responce(f, indexs, label, logs_temp, k, israndom=True):
    length = len(indexs)
    templates = []
    if israndom:
        seed = time.time()
        random.seed(seed)
        random.shuffle(indexs)
        random.seed(seed)
        random.shuffle(logs_temp)

    # # get all templates
    # if length == 1:
    #     f.write(f"---------------------------\n")
    #     f.write(f"cluster {label}: len={length}\n")
    #     f.write(f"---------------------------\n")
    #     print(f"cluster {label}: len={length}")
    #     pass
    # else:
    #     # 按20个一批分批处理
    for i in range(0, len(logs_temp), k):
        batch = logs_temp[i:i+k]
        template_batch = batch_parsing(batch)
        if template_batch != '':
            templates.append(template_batch)

    templates = [postprocessing(template) for template in templates]

    f.write(f"---------------------------\n")
    f.write(f"cluster {label}: len={length}\n")
    print(f"cluster {label}: len={length}")

    # 使用Counter计算频率
    freq = Counter(templates)
    length = len(freq) 
    candidates = freq.most_common(len(freq))
    if length == 0:
        output_tempalte = ''
    elif length == 1 or candidates[0][1] > candidates[1][1]:
        output_tempalte = candidates[0][0]
    else:
        output_tempalte = choose(candidates)
        
    # 打印结果
    for key, value in freq.items():
        f.write(f"{key}: {value}\n")
        print(f"{key}: {value}")
    # for template in templates:
    #     f.write(f"{template}\n")
    #     print(template)
    f.write(f"---------------------------\n")
    return indexs, output_tempalte


def single_dataset_paring(dataset, k):

    # 读取CSV文件
    df = pd.read_csv(
        f'dataset/{dataset}/{dataset}_2k.log_structured_corrected.csv')
    logs = df['Content'].tolist()

    # 对logs分词，得到logs_label，二维列表
    tokenized_logs = [tokenize(log) for log in logs]
    logs_label = []
    for i in range(k):
        logs_label.append([])
    labels = cluster(vectorize(tokenized_logs), k)
    for i, label in enumerate(labels):
        logs_label[label].append(i)

    f = open(f'outputs\k_means\{dataset}.txt', 'w')
    outputs = [None for _ in range(2000)]
    for label, indexs in enumerate(logs_label):
        logs_temp = [logs[i] for i in indexs]
        indexs_after, template = get_responce(f, indexs, label, logs_temp, 50)
        for index in indexs_after:
            outputs[index] = template

    f.close()
    df['Output'] = outputs
    df[['Content', 'EventTemplate', 'Output']].to_csv(
        f'outputs/k_means/initial/{dataset}.csv', index=False)


# main
if __name__ == "__main__":
    datasets = ['BGL', 'HDFS', 'Linux', 'HealthApp', 'OpenStack', 'OpenSSH', 'Proxifier', 'HPC', 'Zookeeper', 'Mac',
                'Hadoop', 'Android', 'Windows', 'Apache', 'Thunderbird', 'Spark']
    cluster_nums = [132, 14, 143, 71, 56, 180, 14, 51, 54, 350, 115, 189, 57, 6, 194, 38]
                 # [120, 14, 116, 75, 43,  26,  8, 46, 50, 341, 114, 158, 50, 6, 149, 36]
    for index, dataset in enumerate(datasets):
        k = cluster_nums[index]
        single_dataset_paring(dataset, k)
    # for dataset in datasets:
    #     single_dataset_paring(dataset, k)
