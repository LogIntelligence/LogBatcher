from concurrent.futures import ThreadPoolExecutor
import json
import os
import time
import pandas as pd
from tqdm import tqdm
from utils.evaluator import evaluate
from utils.cluster import Cluster,tokenize, vectorize, cluster, reassign_clusters
from utils.parser import Cluster_Parser
from utils.sample_byword import matches_template


def batch_logs_paring(f, batch_inputs, parser, cache_pairs=[], Concurrent=True):
    batch_logs = [input[0] for input in batch_inputs]
    batch_templates = [input[2] for input in batch_inputs]

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
        c = Cluster(*input, remove_duplicate=True)
        clusters.append(c)

    # Concurrent or not
    if Concurrent:
        templates = []
        with ThreadPoolExecutor(max_workers=16) as executor:
            templates = list(
                tqdm(executor.map(parser.get_responce,[f]*len(clusters), clusters),
                    total=len(clusters)))
        for c, template in zip(clusters, templates):
            template_exist = any(pair[1] == template for pair in cache_pairs)
            if not template_exist and template != '<*>' and template.strip() != '':
                cache_pairs.append([c.logs[0], template])
            for index in c.indexs:
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
    # return cache_pairs, outputs
    return cache_pairs, outputs


def single_dataset_paring(dataset, output_dir, parser):
    print(f'Parsing {dataset}...')

    # initialize
    # df = pd.read_csv(f'dataset/{dataset}/{dataset}_2k.log_structured_corrected.csv')
    df = pd.read_csv(f'dataset/{dataset}/{dataset}_full.log_structured.csv')
    logs = df['Content'].tolist()
    template = df['EventTemplate'].tolist()
    
    length = len(logs)

    # output file
    os.makedirs(output_dir, exist_ok=True)
    f = open(output_dir + f'{dataset}.txt', 'w')

    # initialize
    batch_size = 2000
    cache_pairs = []

    outputs = [None for _ in range(length)]

    unparsed_inputs = []

    count_parsed = 0
    for i in range(length//batch_size):
        print(f'Parsing {dataset} batch {i+1}...')
        current_batch = logs[i*batch_size:(i+1)*batch_size]
        current_batch_templates = template[i*batch_size:(i+1)*batch_size]

        count_match = 0
        for index, log in enumerate(current_batch):
            parsed = False
            for cache_pair in cache_pairs:
                match_result = matches_template(log, cache_pair)
                if match_result != None:
                    outputs[index + i*batch_size] = match_result
                    count_match += 1
                    count_parsed += 1
                    parsed = True
                    break

            if not parsed:
                unparsed_inputs.append((log, index + i*batch_size, current_batch_templates[index]))

        print(
            f"Matched {count_match} logs in cache, still {len(unparsed_inputs)} logs need to be parsed. {count_parsed} logs have been parsed in total.")

        if len(unparsed_inputs) >= batch_size:
            cache_pairs, batch_outputs = batch_logs_paring(f, unparsed_inputs, parser, cache_pairs=cache_pairs, Concurrent=True)
            for input, output in zip(unparsed_inputs, batch_outputs):
                outputs[input[1]] = output
                count_parsed += 1
            unparsed_inputs = []
        
    if unparsed_inputs:
        print(f'{len(unparsed_inputs)} logs left and need to be parsed.')
        cache_pairs, batch_outputs = batch_logs_paring(f, unparsed_inputs, parser, cache_pairs=cache_pairs, Concurrent=True)
        for input, output in zip(unparsed_inputs, batch_outputs):
            outputs[input[1]] = output
            count_parsed += 1
        unparsed_inputs = []

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
    output_dir = 'outputs/parser/Test_full_dataset/'
    datasets = ['BGL']
    with open('config.json', 'r') as f:
        config = json.load(f)
    parser = Cluster_Parser(config)

    for index, dataset in enumerate(datasets):
        t1 = time.time()
        single_dataset_paring(dataset, output_dir, parser)
        t2 = time.time()
        print(f"finish parsing {dataset}, takes {t2-t1}")
