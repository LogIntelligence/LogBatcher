from collections import Counter
import math
from utils.sample_byword import extract_variables
import pandas as pd


# entropy based sampling
# messages.append({"role": "user", "content": '2017-07-02 15:46:41.445 ksfetch[32435/0x7fff79824000] [lvl=2] main() ksfetch fetching URL (<NSMutableURLRequest: 0x1005110b0> { URL: https://tools.google.com/service/update2?cup2hreq=53f725cf03f511fab16f19e789ce64aa1eed72395fc246e9f1100748325002f4&cup2key=7:1132320327 }) to folder:/tmp/KSOutOfProcessFetcher.YH2CjY1tnx/download'})
# messages.append({"role": "assistant", "content": '`{{timestamp}} ksfetch[{{process_and_thread_id}}] [lvl={{log_level}}] main() ksfetch fetching URL (<NSMutableURLRequest: {{request_id}}> { URL: {{request_url}} }) to folder:{{folder_path}}`'})

def calculate_entropy(lst):
    # 计算列表中每个元素出现的频率

    # list to str
    # print(''.join(lst))

    counter = Counter(lst)
    probs = [count / len(lst) for count in counter.values()]

    # 计算信息熵
    entropy = -sum(p * math.log2(p) for p in probs)

    return entropy

def entropy_calculate(inputs, shot, type = 'pair'):
    if type == 'pair':
        entropies = [(pair, calculate_entropy(list(pair[0]) + list(pair[1]))) for pair in inputs]
    elif type == 'varaible':
        entropies = [(input, calculate_entropy(extract_variables(input[0], input[1]))) for input in inputs]
    # sort by entropy
    sorted_pairs = sorted(entropies, key=lambda x: x[1], reverse=True)

    # select top-k pairs
    selected_pairs = sorted_pairs[:shot]
    return [pair for pair in selected_pairs]

def sample_based_on_entropy(dataset, shot = 5):
    # sample log-template pairs from other datasets
    datasets = ['BGL', 'HDFS', 'Linux', 'HealthApp', 'OpenStack', 'OpenSSH', 'Proxifier', 'HPC', 'Zookeeper', 'Mac',
            'Hadoop', 'Android', 'Windows', 'Apache', 'Thunderbird', 'Spark']
    datasets.remove(dataset)
    pairs =[]
    templates = []
    for d in datasets:
        df = pd.read_csv(f'dataset\{d}\{d}_2k.log_structured_corrected.csv')
        list1 = df['Content'].tolist()
        list2 = df['EventTemplate'].tolist()
        for log, template  in zip(list1, list2):
            if template not in templates:
                pairs.append((log, template, d))
                templates.append(template)

    # filter
    # for pair in pairs:
    #     if len(pair[0]) >= 500:
    #         pairs.remove(pair)
    return entropy_calculate(pairs, shot, type = 'pair')