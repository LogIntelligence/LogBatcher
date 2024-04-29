from concurrent.futures import ThreadPoolExecutor
import json
import os
import pandas as pd
from tqdm import tqdm
from utils.evaluator import evaluate
from utils.cluster import Cluster,tokenize, vectorize, cluster, reassign_clusters
from utils.parser import Cluster_Parser


def batch_logs_paring(f, batch_logs, batch_templates, cache_pairs, parser, Concurrent = True):
    # tokenize -> vectorize -> cluster -> reassign_clusters
    tokenized_logs = [tokenize(log) for log in batch_logs]
    labels, cluster_nums = cluster(vectorize(tokenized_logs))
    labels, cluster_nums = reassign_clusters(labels, cluster_nums, tokenized_logs)


    outputs = [None for _ in range(len(batch_logs))]
    
    inputs = []
    for i in range(cluster_nums):
        inputs.append([-1, [], [], '']) # label, logs, indexs, oracle_template
    for i, label in enumerate(labels):
        inputs[label][0] = label
        inputs[label][1].append(batch_logs[i])
        inputs[label][2].append(i)
        if inputs[label][3] == '':
            inputs[label][3] = batch_templates[i]
    
    clusters = []
    for input in inputs:
        c = Cluster(*input)
        clusters.append(c)

    # Concurrent or not
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
        for c in tqdm(clusters):
            template = parser.get_responce(f, c, cache_pairs)
            template_exist = any(pair[1] == template for pair in cache_pairs)
            if not template_exist and template != '<*>' and template.strip() != '':
                cache_pairs.append([c.logs[0],template])
            for index in c.indexs:
                outputs[index] = template
    return cache_pairs, outputs


def single_dataset_paring(dataset, output_dir, parser, Concurrent = True):
    print(f'Parsing {dataset}...')

    # initialize
    df = pd.read_csv(f'dataset/{dataset}/{dataset}_2k.log_structured_corrected.csv')
    logs = df['Content'].tolist()
    template = df['EventTemplate'].tolist()
    
    # output file
    os.makedirs(output_dir, exist_ok=True)
    f = open(output_dir + f'{dataset}.txt', 'w')

    # 将logs切分成10个batch
    batch_size = 200
    cache_pairs = []
    outputs = []
    for i in range(2000//batch_size):
        print(f'Parsing {dataset} batch {i+1}...')
        batch_logs = logs[i*batch_size:(i+1)*batch_size]
        batch_templates = template[i*batch_size:(i+1)*batch_size]
        cache_pairs, batch_outputs = batch_logs_paring(f, batch_logs, batch_templates, cache_pairs, parser, Concurrent = False)
        outputs.extend(batch_outputs)

    # write to file
    f.write(f"---------cache_pairs---------\n")
    for pair in cache_pairs:
        f.write(f"{pair[1]}\n")
    f.close()
    df['Output'] = outputs
    df[['Content', 'EventTemplate', 'Output']].to_csv(output_dir+ f'{dataset}.csv', index=False)
    evaluate(output_dir + f'{dataset}.csv', dataset)


# main
if __name__ == "__main__":
    datasets = ['BGL', 'HDFS', 'Linux', 'HealthApp', 'OpenStack', 'OpenSSH', 'Proxifier', 'HPC', 'Zookeeper', 'Mac', 'Hadoop', 'Android', 'Windows', 'Apache', 'Thunderbird', 'Spark']
    output_dir = 'outputs/parser/Test_batch/'
    datasets = ['BGL', 'HDFS', 'Linux', 'HealthApp', 'OpenStack', 'OpenSSH', 'Proxifier', 'HPC', 'Zookeeper', 'Mac', 'Hadoop', 'Android', 'Windows', 'Apache', 'Thunderbird']
    with open('config.json', 'r') as f:
        config = json.load(f)
    parser = Cluster_Parser(config)

    for index, dataset in enumerate(datasets):
        single_dataset_paring(dataset, output_dir, parser, Concurrent=False)
