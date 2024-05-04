from concurrent.futures import ThreadPoolExecutor
import json
import os
import time
import pandas as pd
from tqdm import tqdm
from utils.evaluator import evaluate
from utils.cluster import Cluster,tokenize, vectorize, cluster, reassign_clusters
from utils.parser import Cluster_Parser
from evaluate import evaluate_all_datasets


def single_dataset_paring(dataset, output_dir, parser, Concurrent = True):
    print(f'Parsing {dataset}...')

    # initialize
    df = pd.read_csv(f'dataset/{dataset}/{dataset}_2k.log_structured_corrected.csv')
    logs = df['Content'].tolist()

    # tokenize -> vectorize -> cluster -> reassign_clusters
    tokenized_logs = [tokenize(log) for log in logs]
    labels, cluster_nums = cluster(vectorize(tokenized_logs))
    num = cluster_nums
    labels, cluster_nums = reassign_clusters(labels, cluster_nums, tokenized_logs)

    # output file
    os.makedirs(output_dir, exist_ok=True)
    f = open(output_dir + f'{dataset}.txt', 'w')

    outputs = [None for _ in range(2000)]
    tmps_list = [None for _ in range(2000)]
    
    inputs = []
    for i in range(cluster_nums):
        inputs.append([-1, [], [], '']) # label, logs, indexs, oracle_template
    for i, label in enumerate(labels):
        inputs[label][0] = label
        inputs[label][1].append(logs[i])
        inputs[label][2].append(i)
        if inputs[label][3] == '':
            inputs[label][3] = df['EventTemplate'][i]
    
    clusters = []
    for input in inputs:
        c = Cluster(*input, remove_duplicate= False)
        clusters.append(c)

    # Concurrent or not
    # if Concurrent, then the parsing process will be faster but we can't do something like cache parsing
    if Concurrent:
        templates = []
        with ThreadPoolExecutor(max_workers=16) as executor:
            templates = list(
                tqdm(executor.map(parser.get_responce,[f]*len(clusters), clusters),
                    total=len(clusters)))
        for label, template in enumerate(templates):
            for index in inputs[label][2]:
                outputs[index] = template
    else:
        clusters = sorted(clusters, key=lambda cluster: len(cluster.indexs), reverse=True)
        cache_pairs = []
        for c in tqdm(clusters):
            tmps, template = parser.get_responce(f, c, cache_pairs)
            template_exist = any(pair[1] == template for pair in cache_pairs)
            if not template_exist and template != '<*>' and template.strip() != '':
                cache_pairs.append([c.logs[0],template])
            for index in c.indexs:
                outputs[index] = template
                tmps_list[index] = '\n'.join(tmps)

    # write to file
    f.close()
    df['Tmps'] = tmps_list
    df['Output'] = outputs
    df[['Content', 'EventTemplate', 'Tmps','Output']].to_csv(output_dir+ f'{dataset}.csv', index=False)
    evaluate(output_dir + f'{dataset}.csv', dataset)


# main
if __name__ == "__main__":
    time.sleep(30 * 60)
    datasets = ['BGL', 'HDFS', 'Linux', 'HealthApp', 'OpenStack', 'OpenSSH', 'Proxifier', 'HPC', 'Zookeeper', 'Mac', 'Hadoop', 'Android', 'Windows', 'Apache', 'Thunderbird', 'Spark']
    datasets = ['BGL', 'HDFS', 'HealthApp', 'OpenStack', 'OpenSSH', 'Proxifier', 'HPC',
                'Zookeeper', 'Mac', 'Hadoop', 'Android', 'Windows', 'Apache', 'Thunderbird', 'Spark']
    theme = 'Test3'
    output_dir = f'outputs/parser/{theme}/'
    with open('config.json', 'r') as f:
        config = json.load(f)
    parser = Cluster_Parser(config)

    for index, dataset in enumerate(datasets):
        single_dataset_paring(dataset, output_dir, parser, Concurrent=False)
    evaluate_all_datasets(theme,send_email=True)
