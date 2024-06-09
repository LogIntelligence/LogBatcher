import argparse
import json
import re
import time
import pandas as pd
from tqdm import tqdm
from utils.cluster import Cluster,tokenize, vectorize, cluster, reassign_clusters
from utils.parser import Cluster_Parser
from utils.evaluator import evaluate
from utils.sample import sample_from_clusters
from utils.util import generate_logformat_regex, log_to_dataframe
from tqdm import tqdm

from utils.matching import matches_template


def single_dataset_paring(dataset, log_format, output_dir, parser, shot, candidate, batch_size, chunk_size , sample_method = 'dpp', log_file_format = 'structured', data_type = '2k'):

    # Initializing
    t0 = time.time()
    if data_type == '2k':
        structured_log_file = f'dataset/{dataset}/{dataset}_2k.log_structured_corrected.csv'
    elif data_type == 'full':
        structured_log_file = f'dataset/{dataset}/{dataset}_full.log_structured.csv'
    else:
        raise ValueError('data_type should be 2k or full')
    
    if log_file_format == 'structured':
        df = pd.read_csv(structured_log_file)
        logs = df['Content'].tolist()
    elif log_file_format == 'raw':
        log_file = f'dataset/{dataset}/{dataset}.log'
        with open(log_file, 'r') as f:
            log_raws = f.readlines()
        headers, regex = generate_logformat_regex(log_format)
        logs = log_to_dataframe(log_file, regex, headers, len(log_raws))
    else:
        raise ValueError('log_file_format should be structured or raw')

    outputs = [None for _ in range(len(logs))]
    cache_pairs = {}
    log_chunk = []
    log_chunk_index = []
    cache_sort_step = len(logs) / 100
    print(f'Parsing {len(logs)} logs in dataset {dataset}...')

    # Parsing
    t1 = time.time()
    for index, log in enumerate(tqdm(logs)):

        # Cache Sorting
        if index % cache_sort_step == 0 and len(cache_pairs) != 0:
            cache_pairs = dict(sorted(cache_pairs.items(), key=lambda item: item[1][1], reverse=True))

        # Cache Matching
        for template, value_f in cache_pairs.items():
            match_result = matches_template(log, [value_f[0], template])
            if match_result != None and match_result in cache_pairs:
                cache_pairs[match_result][1] += 1
                outputs[index] = match_result
                break
        if outputs[index] == None:
            log_chunk.append(log)
            log_chunk_index.append(index)

        # Parsing with LLM
        if len(log_chunk) == chunk_size or (len(log_chunk)!=0 and index == len(logs) - 1):
            # parsing start
            # tokenize -> vectorize -> cluster -> reassign_clusters
            tokenized_logs = [tokenize(log) for log in log_chunk]
            labels, cluster_nums = cluster(vectorize(tokenized_logs))
            labels, cluster_nums = reassign_clusters(labels, cluster_nums, tokenized_logs)
            
            # store the logs in each cluster and sort them by the number of logs in each cluster
            inputs = []
            clusters = []
            for i in range(cluster_nums):
                inputs.append([-1, [], [], '']) # label, logs, indexs, oracle_template
            for i, label in zip(log_chunk_index, labels):
                inputs[label][0] = label
                inputs[label][1].append(logs[i])
                inputs[label][2].append(i)
                inputs[label][3] = ''
            for input in inputs:
                c = Cluster(*input, remove_duplicate=True,remain_num=batch_size, sample_method=sample_method)
                clusters.append(c)

            clusters = sorted(clusters, key=lambda cluster: len(cluster.indexs), reverse=True)
        

            # parse each cluster
            for index, c in enumerate(clusters):
                template, c, new_cluster = parser.get_responce( c, cluster_nums, cache_pairs, [], 0)

                # update clusters
                if new_cluster != None:
                    clusters.append(new_cluster)
                    cluster_nums += 1

                if template not in cache_pairs and template.replace('<*>','').replace(' ','') != '':
                    cache_pairs[template] = [c.logs[0], 0]
                
                for index in c.indexs:
                    outputs[index] = template
            log_chunk = []
            log_chunk_index = []
    
    # Result
    t2 = time.time()
    print(f'initial time: {t1 - t0}')
    print(f'parsing time: {t2 - t1}')
    print(f'idetified templates: {len(set(outputs))}')
    # write to file
    df_new = pd.DataFrame()
    df_new['Content'] = logs
    df_new['EventTemplate'] = outputs
    output_log_file = output_dir + f'{dataset}_{data_type}.log_structured.csv'
    df_new.to_csv(output_log_file, index=False)
    with open(output_log_file.replace('structured.csv', 'templates.txt'), 'w') as f:
        for template, value_f in cache_pairs.items():
            f.write("%7d:%s\n" % (value_f[1], template))
    
    evaluate(output_log_file, structured_log_file, dataset)


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
    parser.add_argument('--chunk_size', type=int, default=2000,
                        help='Size of logs in a chunk')
    # 解析命令行参数
    args = parser.parse_args()
    # 调用处理函数
    return args


if __name__ == "__main__":
    args = set_args()
    datasets = ['BGL', 'HDFS', 'HealthApp', 'OpenStack', 'OpenSSH', 'HPC', 'Zookeeper',
                'Mac', 'Hadoop', 'Android', 'Windows', 'Apache', 'Thunderbird', 'Spark', 'Linux']
    # dataset_format = {
    #     'BGL':'<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>',
    #     'HDFS':'<Date> <Time> <Pid> <Level> <Component>: <Content>',
    #     'OpenStack':'<Timestamp> <Node> <Component> <Level> <Content>',
    #     'Zookeeper':'<Date> <Time> <Level> \[<Node>:<Component>@<Id>\] - <Content>',
    #     'OpenSSH':'<Date> <Time> <Pid> <Level> <Component> <Content>',
    #     'Apache':'<Content>',
    #     }
    datasets = ['HDFS']
    model = args.model
    
    theme = f"LogBatcher_{args.shot}shot_{args.candidate}candidate_{args.batch_size}batchsize_{args.chunk_size}chunksize_full_time"

    output_dir = f'outputs/parser/{theme}/'
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    # else:
    #     print(f'{output_dir} already exists.\nresults is here: {output_dir}')
    #     exit()
    with open('config.json', 'r') as f:
        config = json.load(f)
    config['model'] = args.model
    
    for index, dataset in enumerate(datasets):
        parser = Cluster_Parser(theme, config)
        single_dataset_paring(
            dataset=dataset, 
            # log_format = dataset_format[dataset],
            log_format = '',
            output_dir=output_dir, 
            parser=parser, 
            shot=args.shot,
            candidate=args.candidate,
            batch_size=args.batch_size,
            chunk_size=args.chunk_size,
            sample_method = args.sample_method,
            log_file_format = 'structured',
            data_type = 'full'
        )
        print('time cost by llm: ', parser.time_consumption_llm)
