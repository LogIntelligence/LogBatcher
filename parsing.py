from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import os
import random
import pandas as pd
import re
from sklearn.cluster import KMeans, DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from openai import OpenAI
import httpx
from tenacity import retry, stop_after_attempt, wait_random_exponential
from tqdm import tqdm
from evaluator import evaluate
from post_process import correct_single_template


class clusters:
    def __init__(self, label, logs, indexs, ground_truth):
        self.label = label
        self.logs = logs
        self.indexs = indexs
        self.ground_truth = ground_truth
    
    def remove_duplicate(self):
        self.logs = list(set(self.logs))

def tokenize(log_content, tokenize_pattern=r'[ ,]'):
    words = re.split(tokenize_pattern, log_content)
    for index, word in enumerate(words):
        if '=' in word:
            words[index] = word.split('=')[0]
        if re.search(r'\d', word):
            words[index] = ''
    words = [word for word in words if word]   # remove null
    return words


def vectorize(tokenized_logs):
    vectorizer = TfidfVectorizer(tokenizer=lambda x: x, lowercase=False)
    return vectorizer.fit_transform(tokenized_logs)


def cluster(vectorized_logs, num_clusters='10', cluster_method='kmeans'):
    if cluster_method == 'kmeans':
        cluster = KMeans(n_clusters=num_clusters)
    if cluster_method == 'dbscan':
        cluster = DBSCAN(eps=0.1, min_samples=5)
    cluster.fit(vectorized_logs)
    labels = cluster.labels_
    cluster_nums = max(labels) + 1
    return labels, cluster_nums
    

def reassign_clusters(labels, cluster_nums, tokenized_logs):
    mergerd_logs = []
    for tokenized_log in tokenized_logs:
        mergerd_logs.append(' '.join(tokenized_log))

    for i in range(len(labels)):
        if labels[i] == -1:
            for j in range(i+1, len(labels)):
                if labels[j] == -1 and mergerd_logs[i] == mergerd_logs[j]:
                    labels[j] = cluster_nums
            labels[i] = cluster_nums
            cluster_nums += 1
    return labels, cluster_nums

class Parser:
    def __init__(self, api_key, model='gpt-3.5-turbo-0613', using_proxy=True, cluster_method='dbscan', batch_num=50):
        self.api_key = api_key
        self.model = model
        self.cluster_method = cluster_method
        self.batch_num = batch_num
        self.random = True
        self.instruction_batch = '''You will be provided with some log messages. You should check if the giving log messages share the same template. If so, abstract variables with `{{placeholders}}` to extract the corresponding template.
        Print the input log's template delimited by backticks.'''
        self.instruciton_one_log = '''You will be provided with a log message delimited by backticks. You must abstract variables with `{{placeholders}}` to extract the corresponding template.
        Print the input log's template delimited by backticks.'''
        if using_proxy:
            self.client = OpenAI(
                # 3.5 https://4.0.996444.icu/v1
                base_url="https://oneapi.xty.app/v1",  # 中转url
                api_key=api_key,                      # api_key
                http_client=httpx.Client(
                    proxies="http://127.0.0.1:7890"  # 代理地址
                ),
            )
        else:
            self.client = OpenAI(
                base_url="https://oneapi.xty.app/v1", api_key=api_key)

    # @backoff.on_exception(backoff.expo, (openai.APIStatusError, openai.InternalServerError), max_tries=5)
    @retry(wait=wait_random_exponential(min=1, max=30), stop=stop_after_attempt(5))
    def chat(self, messages):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.0,
        )
        return response.choices[0].message.content.strip('\n')
    
    def get_responce(self, f, input):
        label, logs, indexs, ground_truth = input
        length = len(indexs)
        templates = []

        # remove duplicate
        # logs = list(set(logs))

        # remove duplicate conditionally
        # if len(list(set(logs))) == 1:
        #     logs = list(set(logs))
        
        if self.random:
            # seed = time.time()
            seed = 0
            random.seed(seed)
            random.shuffle(logs)

        for i in range(0, len(logs), self.batch_num):
            batch_logs = logs[i:i+self.batch_num]
            # if all logs's length is 1, and not contain any digit, return the log itself
            if all(len(re.split(' ', log)) == 1 and not any(char.isdigit() for char in log) for log in batch_logs):
                return batch_logs[0]

            messages = []
            if len(batch_logs) == 1:
                messages.append({"role": "system", "content": self.instruciton_one_log})
            else:
                messages.append({"role": "system", "content": self.instruction_batch})
            # batch logs to str
            prompt = ""
            length_prompt = 0
            for log in batch_logs:
                prompt += log + '\n'
                length_prompt += len(log)
            if length_prompt > 4096:
                prompt = ""
                for log in batch_logs[:5]:
                    prompt += log + '\n'
            messages.append({"role": "user", "content": prompt.strip('\n')})
            answer = self.chat(messages)
            template =  postprocessing(answer , isafter=False)
            if template != '':
                templates.append(template)

            f.write(f"---------------------------\n")
            f.write(f"cluster {label}: len={length}\n")
            f.write(f"{ground_truth} (ground truth)\n")
            # print(f"cluster {label}: len={length}")

            final_tempalte, freq = choose(templates)
                
            # 打印结果
            for key, value in freq.items():
                f.write(f"{key}: {value}\n")
                # print(f"{key}: {value}")
            f.write(f"---------------------------\n")
            return final_tempalte




def postprocessing(response, isafter = False):

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
        boolean = {'true', 'false'}
        default_strings = {'null', 'root', 'admin'}
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

    


def choose(list):

    # majority vote
    freq = Counter(list)
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


def single_dataset_paring(dataset, output_dir, k = 10, cluster_method='kmeans', isConcurrent = True):
    print(f'Parsing {dataset}...')
    parser = Parser(
        api_key='sk-j9uJ8yuwjNL2fgre21C203D01f1546EcA389F514C94829Ff')
    # sk-4iNXitFaZ2fOJdtq8b89D44229Ec4fAd9c0f00D4087c6541 4.0
    # sk-6ZwLXFPGK6pVfKrKFdDcA5D2B25f480285Be7a17A0385d8b 3.5
    # load dataset
    df = pd.read_csv(f'dataset/{dataset}/{dataset}_2k.log_structured_corrected.csv')
    logs = df['Content'].tolist()

    # tokenize
    tokenized_logs = [tokenize(log) for log in logs]
    
    # cluster 1st
    labels, cluster_nums = cluster(vectorize(tokenized_logs), k, cluster_method)
    
    # reassign_clusters
    labels, cluster_nums = reassign_clusters(labels, cluster_nums, tokenized_logs)

    # output file
    os.makedirs(output_dir, exist_ok=True)
    f = open(output_dir + f'{dataset}.txt', 'w')

    outputs = [None for _ in range(2000)]
    
    inputs = []
    for i in range(cluster_nums):
        inputs.append([-1, [], [], '']) # label, logs, indexs, ground_truth
    for i, label in enumerate(labels):
        inputs[label][0] = label
        inputs[label][1].append(logs[i])
        inputs[label][2].append(i)
        if inputs[label][3] == '':
            inputs[label][3] = df['EventTemplate'][i]
    
    # Concurrent or not
    if isConcurrent:
        templates = []
        with ThreadPoolExecutor(max_workers=16) as executor:
            templates = list(
                tqdm(executor.map(parser.get_responce,[f]*len(inputs), inputs),
                    total=len(inputs)))
        for label, template in enumerate(templates):
            for index in inputs[label][2]:
                outputs[index] = template
    else:
        for label in range(cluster_nums):
            template = parser.get_responce(f, inputs[label])
            for index in inputs[label][2]:
                outputs[index] = template

    # write to file
    f.close()
    df['Output'] = outputs
    df[['Content', 'EventTemplate', 'Output']].to_csv(output_dir+ f'{dataset}.csv', index=False)
    evaluate(output_dir + f'{dataset}.csv', dataset)


# main
if __name__ == "__main__":
    datasets = ['BGL', 'HDFS', 'Linux', 'HealthApp', 'OpenStack', 'OpenSSH', 'Proxifier', 'HPC', 'Zookeeper', 'Mac',
                'Hadoop', 'Android', 'Windows', 'Apache', 'Thunderbird', 'Spark']
    # datasets = ['Linux']
    # datasets = ['Thunderbird', 'Spark']
    cluster_nums = [132, 14, 143, 71, 56, 180, 14, 51, 54, 350, 115, 189, 57, 6, 194, 38]
    cluster_nums = [120, 14, 116, 75, 43,  26,  8, 46, 50, 341, 114, 158, 50, 6, 149, 36]
                 # [120, 14, 116, 75, 43,  26,  8, 46, 50, 341, 114, 158, 50, 6, 149, 36]
    output_dir = 'outputs/parser/0613_0shot_RDC/'
    for index, dataset in enumerate(datasets):
        k = cluster_nums[index]
        single_dataset_paring(dataset, output_dir, cluster_method='dbscan')
    
    # single_dataset_paring('Linux', cluster_method='dbscan')
