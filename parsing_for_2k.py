from concurrent.futures import ThreadPoolExecutor
import json
import os
import time
import pandas as pd
from tqdm import tqdm
from utils.evaluator import evaluate
from utils.cluster import Cluster,tokenize, vectorize, cluster, reassign_clusters
from utils.parser import Cluster_Parser
from evaluate import evaluate_all_datasets, evaluate_single_dataset
from utils.sample import sample_from_clusters
from utils.prune import prune_from_cluster

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
        c = Cluster(*input, remove_duplicate= True, remain_num=5)
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
            #ablation: without caching
            # tmps, template = parser.get_responce(f, c, [])
            tmps, template, has_result = parser.get_responce(f, c, cache_pairs)

            # Matching & Pruning
            if has_result:
                c, new_c =  prune_from_cluster(template, c, cluster_nums)
                if new_c != None:
                    clusters.append(new_c)
                    cluster_nums += 1
                    f.write(f"---------------------------\n")
                    f.write(f"cluster {c.label}: form a new cluster\n")
                    f.write(f"---------------------------\n")


            # update cache
            template_exist = any(pair[1] == template for pair in cache_pairs)
            if not template_exist and template != '<*>' and template.strip() != '':
                cache_pairs.append([c.logs[0],template])

            for index in c.indexs:
                outputs[index] = template
                tmps_list[index] = '\n'.join(tmps)

    # write to file
    f.close()
    df['Tmps'] = tmps_list
    df['EventTemplate'] = outputs
    df[['Content','Tmps','EventTemplate']].to_csv(
        output_dir + f'{dataset}_2k.log_structured.csv', index=False)
    evaluate_single_dataset(output_dir + f'{dataset}_2k.log_structured.csv', dataset)


# main
if __name__ == "__main__":
    datasets = ['BGL', 'HDFS', 'HealthApp', 'OpenStack', 'OpenSSH', 'HPC', 'Zookeeper', 'Mac', 'Hadoop', 'Android', 'Windows', 'Apache', 'Thunderbird', 'Spark', 'Linux']
    datasets = ['BGL', 'HDFS', 'HealthApp']

    theme = 'Test_10shot_with_pruning'
    output_dir = f'outputs/parser/{theme}/'
    with open('config.json', 'r') as f:
        config = json.load(f)
    parser = Cluster_Parser(config)

    for index, dataset in enumerate(datasets):
        single_dataset_paring(dataset, output_dir, parser, Concurrent=False)
    # evaluate_all_datasets(theme,send_email=True)
