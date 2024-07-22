import argparse
from collections import Counter
import json
import os
import re
import string
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

def verify_template(template):
    template = template.replace("<*>", "")
    template = template.replace(" ", "")
    return any(char not in string.punctuation for char in template)


def single_dataset_paring(dataset, log_format, output_dir, parser, batch_size, chunk_size , sample_method = 'dpp', log_file_format = 'structured', data_type = 'full', debug=True):

    # Initializing
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
    identified_templates_num = 0
    cache_pairs = {}
    log_chunk = []
    log_chunk_index = []
    cache_sort_step = len(logs) // 100
    print(f'Parsing {len(logs)} logs in dataset {dataset}...')

    # Parsing
    t1 = time.time()
    iterable = tqdm(enumerate(logs), total=len(logs), unit="log")
    for index, log in iterable:

        # Cache Sorting
        if (index % cache_sort_step) == 0 and len(cache_pairs) != 0:
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
            print(f'Parsing {len(log_chunk)} logs...') if debug else None
            # tokenize -> vectorize -> cluster -> reassign_clusters
            tokenized_logs = [tokenize(log) for log in log_chunk]
            labels, cluster_nums = cluster(vectorize(tokenized_logs))
            labels, cluster_nums = reassign_clusters(labels, cluster_nums, tokenized_logs)

            # create clusters
            clusters = [None for _ in range(cluster_nums)]
            for index, label in enumerate(labels):
                if clusters[label] is None:
                    clusters[label] = Cluster()
                clusters[label].append_log(log_chunk[index], log_chunk_index[index])

            # sorting
            clusters = sorted(clusters, key=lambda cluster: len(cluster.logs), reverse=True)

            # batching
            [cluster.batching(batch_size, sample_method) for cluster in clusters]
        
            # parsing
            for index, old_cluster in enumerate(clusters):
                template, old_cluster, new_cluster = parser.get_responce(old_cluster, cache_pairs)

                if debug:
                    print('=' * 20)
                    print(f'New cluster processed, {identified_templates_num + 1} templates identified till now:')
                    print(f'Refer Log: {old_cluster.logs[0]}')
                    print(f'Output Template: {template}')

                # update clusters
                if new_cluster.size != 0:
                    new_cluster.batching(batch_size, sample_method)
                    clusters.append(new_cluster)
                    cluster_nums += 1

                if template not in cache_pairs and verify_template(template):
                    identified_templates_num += 1
                    cache_pairs[template] = [old_cluster.logs[0], 0]
                
                for index in old_cluster.indexs:
                    outputs[index] = template
            log_chunk = []
            log_chunk_index = []
    
    
    # Result
    t2 = time.time()
    print(f'parsing time: {t2 - t1}')
    print(f'idetified templates: {len(set(outputs))}')

    # output logs
    output_log_file = output_dir + f'{dataset}_{data_type}.log_structured.csv'
    df = pd.DataFrame({'Content': logs, 'EventTemplate': outputs})
    df.to_csv(output_log_file, index=False)

    # output templates
    counter = Counter(outputs)
    items = list(counter.items())
    items.sort(key=lambda x: x[1], reverse=True)
    output_template_file = output_dir + f'{dataset}_{data_type}.template_structured.csv'
    template_df = pd.DataFrame(items, columns=['EventTemplate', 'Occurrence'])
    template_df['EventID'] = [f"E{i + 1}" for i in range(len(template_df))]
    template_df[['EventID', 'EventTemplate', 'Occurrence']].to_csv(output_template_file, index=False)

    # Save time cost
    time_cost_file = output_dir + 'time_cost.json'
    time_table = {}
    if os.path.exists(time_cost_file):
        with open(time_cost_file, 'r') as file:
            time_table = json.load(file)
    time_table[dataset] = {
        'InvocatingTime': parser.time_consumption_llm.__round__(3),
        'ParsingTime': (t2 - t1).__round__(3)
    }
    parser.time_consumption_llm = 0
    with open(time_cost_file, 'w') as file:
        json.dump(time_table, file)


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo-0125',
                        help='use which model to parse the log.')
    parser.add_argument('--batch_size', type=int, default=10, 
                        help='The size of a batch')
    parser.add_argument('--sample_method', type=str, default='dpp',
                        help='Sample method: dpp, random, similar.')
    parser.add_argument('--chunk_size', type=int, default=2000,
                        help='Size of logs in a chunk')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = set_args()
    datasets = ['BGL', 'HDFS', 'HealthApp', 'OpenStack', 'OpenSSH', 'HPC', 'Zookeeper',
                'Mac', 'Hadoop', 'Android', 'Windows', 'Apache', 'Thunderbird', 'Spark', 'Linux', 'proxifier']
    datasets = ['Thunderbird']
    
    datasets_format = {
        'HDFS': '<Date> <Time> <Pid> <Level> <Component>: <Content>',
        'Hadoop': '<Date> <Time> <Level> \[<Process>\] <Component>: <Content>',
        'Spark': '<Date> <Time> <Level> <Component>: <Content>',
        'Zookeeper': '<Date> <Time> - <Level>  \[<Node>:<Component>@<Id>\] - <Content>',
        'BGL': '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>',
        'HPC': '<LogId> <Node> <Component> <State> <Time> <Flag> <Content>',
        'Thunderbird': '<Label> <Timestamp> <Date> <User> <Month> <Day> <Time> <Location> <Component>(\[<PID>\])?: <Content>',
        'Windows': '<Date> <Time>, <Level>                  <Component>    <Content>',
        'Linux': '<Month> <Date> <Time> <Level> <Component>(\[<PID>\])?: <Content>',
        'Android': '<Date> <Time>  <Pid>  <Tid> <Level> <Component>: <Content>',
        'HealthApp': '<Time>\|<Component>\|<Pid>\|<Content>',
        'Apache': '\[<Time>\] \[<Level>\] <Content>',
        'Proxifier': '\[<Time>\] <Program> - <Content>',
        'OpenSSH': '<Date> <Day> <Time> <Component> sshd\[<Pid>\]: <Content>',
        'OpenStack': '<Logrecord> <Date> <Time> <Pid> <Level> <Component> \[<ADDR>\] <Content>',
        'Mac': '<Month>  <Date> <Time> <User> <Component>\[<PID>\]( \(<Address>\))?: <Content>'
    }
    
    theme = f"LogBatcher_full_{args.batch_size}batchsize_{args.chunk_size}chunksize_{args.model.replace('/','_')}_{args.sample_method}_sampling"
    output_dir = f'outputs/parser/{theme}/'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # load api key
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    parser = Cluster_Parser(args.model, theme, config)
    for index, dataset in enumerate(datasets):
        if os.path.exists(f'{output_dir}{dataset}_full.log_structured.csv'):
            print(f'{dataset} has been parsed, skip it.')
            continue
        single_dataset_paring(
            dataset=dataset, 
            log_format = datasets_format[dataset],
            output_dir=output_dir, 
            parser=parser, 
            batch_size=args.batch_size,
            chunk_size=args.chunk_size,
            sample_method = args.sample_method,
            log_file_format = 'structured',
            data_type = 'full'
        )
        print('time cost by llm: ', parser.time_consumption_llm)
