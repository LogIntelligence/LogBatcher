import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
import pandas as pd
from tqdm import tqdm
from utils.cluster import Cluster,tokenize, vectorize, cluster, reassign_clusters
from utils.parser import Cluster_Parser
from evaluate import evaluate_all_datasets, evaluate_single_dataset
from utils.sample import sample_from_clusters


def single_dataset_paring(dataset, output_dir, parser, shot, candidate, batch_size, Concurrent=True, sample_method = 'dpp'):
    print(f'Parsing {dataset}...')

    # initialize
    df = pd.read_csv(f'dataset/{dataset}/{dataset}_2k.log_structured_corrected.csv')
    logs = df['Content'].tolist()

    # tokenize -> vectorize -> cluster -> reassign_clusters
    tokenized_logs = [tokenize(log) for log in logs]
    labels, cluster_nums = cluster(vectorize(tokenized_logs))
    labels, cluster_nums = reassign_clusters(labels, cluster_nums, tokenized_logs)

    # output file

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
        c = Cluster(*input, remove_duplicate=True,
                    remain_num=batch_size, sample_method=sample_method)
        clusters.append(c)

    # sample from clusters
    sample_pairs = sample_from_clusters(clusters, candidate)

    # ablation: without clustering
    # clusters = []
    # cluster_nums = int(2000 / batch_size)
    # for i in range(cluster_nums):
    #     clusters.append(Cluster(i, logs[i*batch_size:(i+1)*batch_size], [j for j in range(i*batch_size,(i+1)*batch_size)], '', remove_duplicate=True, remain_num=batch_size, sample_method=sample_method))


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
        for index, c in enumerate(clusters):
            print(f"=" * 40)
            print(f"parsing the cluster {index} in {cluster_nums} clusters\nsample log: {c.logs[0]}")
            #ablation: without caching
            # tmps, template = parser.get_responce(f, c, [])
            tmp, template, c, new_cluster = parser.get_responce( c, cluster_nums, cache_pairs, sample_pairs, shot)

            # update clusters
            if new_cluster != None:
                clusters.append(new_cluster)
                cluster_nums += 1

            # update cache
            template_exist = any(pair[1] == template for pair in cache_pairs)
            if not template_exist and template != '<*>' and template.strip() != '':
                cache_pairs.append([c.logs[0],template])

            for index in c.indexs:
                outputs[index] = template
                tmps_list[index] = tmp

    # write to file
    df['Tmps'] = tmps_list
    df['EventTemplate'] = outputs
    df[['Content','Tmps','EventTemplate']].to_csv(
        output_dir + f'{dataset}_2k.log_structured.csv', index=False)
    evaluate_single_dataset(output_dir + f'{dataset}_2k.log_structured.csv', dataset)


def set_args():
    # 定义命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo-0125',
                        help='use which model to parse the log.')
    parser.add_argument('--candidate', type=int, default=32,
                        help='The num of candidate pairs.')
    parser.add_argument('--shot', type=int, default=0,
                        help='The num of demostrations.')
    parser.add_argument('--batch_size', type=int, default=10, 
                        help='The size of a batch')
    parser.add_argument('--sample_method', type=str, default='dpp',
                        help='Sample method: dpp, random, similar.')
    # 解析命令行参数
    args = parser.parse_args()
    # 调用处理函数
    return args


if __name__ == "__main__":
    args = set_args()
    datasets = ['BGL', 'HDFS', 'HealthApp', 'OpenStack', 'OpenSSH', 'HPC', 'Zookeeper',
                'Mac', 'Hadoop', 'Android', 'Windows', 'Apache', 'Thunderbird', 'Spark', 'Linux']

    model = args.model
    module = ''
    if 'gpt' not in model:
        if '/' in model:
            theme = f"LogBatcher_{args.shot}shot_{args.candidate}candidate_{args.batch_size}batchsize_{model.replace('/','_')}"
        else:
            theme = f"LogBatcher_{args.shot}shot_{args.candidate}candidate_{args.batch_size}batchsize_{model}"
    elif module:
        theme = f"LogBatcher_{args.shot}shot_{args.candidate}candidate_{args.batch_size}batchsize_without_{module}"
    else:
        theme = f"LogBatcher_{args.shot}shot_{args.candidate}candidate_{args.batch_size}batchsize"

    output_dir = f'outputs/parser/{theme}/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        print(f'{output_dir} already exists.\nresults is here: {output_dir}')
        exit()
    with open('config.json', 'r') as f:
        config = json.load(f)
    config['model'] = args.model
    parser = Cluster_Parser(theme, config)
    for index, dataset in enumerate(datasets):
        single_dataset_paring(
            dataset=dataset, 
            output_dir=output_dir, 
            parser=parser, 
            shot=args.shot,
            candidate=args.candidate,
            batch_size=args.batch_size,
            Concurrent=False,
            sample_method = args.sample_method
        )
    evaluate_all_datasets(theme,send_email=True)