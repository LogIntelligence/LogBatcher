import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import re
import time
import pandas as pd
from tqdm import tqdm
from utils.cluster import Cluster,tokenize, vectorize, cluster, reassign_clusters
from utils.parser import Cluster_Parser
from evaluate import evaluate_all_datasets, evaluate_single_dataset
from utils.sample import sample_from_clusters
from tqdm import tqdm

from utils.sample_byword import matches_template

def generate_logformat_regex(logformat):
        """ Function to generate regular expression to split log messages
        """
        headers = []
        splitters = re.split(r'(<[^<>]+>)', logformat)
        regex = ''
        for k in range(len(splitters)):
            if k % 2 == 0:
                splitter = re.sub(' +', '\\\s+', splitters[k])
                regex += splitter
            else:
                header = splitters[k].strip('<').strip('>')
                regex += '(?P<%s>.*?)' % header
                headers.append(header)
        regex = re.compile('^' + regex + '$')
        return headers, regex


def log_to_dataframe(log_file, regex, headers, size):
        """ Function to transform log file to contents 
        """
        log_messages = []
        linecount = 0
        with open(log_file, 'r') as file:
            for line in [next(file) for _ in range(size)]:
                try:
                    match = regex.search(line.strip())
                    message = [match.group(header) for header in headers]
                    log_messages.append(message[-1])
                except Exception as e:
                    pass
        return log_messages


def single_dataset_paring(dataset,log_format, output_dir, parser, shot, candidate, batch_size, chunk_size ,Concurrent=True, sample_method = 'dpp'):
    # num = 1000000
    # print(f'Parsing {int(num/1000)}k logs in dataset {dataset}...')
    t0 = time.time()
    # initialize
    # headers, regex = generate_logformat_regex(log_format)
    # log_file = f'dataset/{dataset}/{dataset}.log'
    # logs = log_to_dataframe(log_file, regex, headers, num)
    # df = pd.read_csv(f'dataset/{dataset}/{dataset}_full.log_structured.csv', nrows=num + 10)
    
    df = pd.read_csv(f'dataset/{dataset}/{dataset}_full.log_structured.csv')
    logs = df['Content'].tolist()
    # logs = logs[:num]
    print(f'Parsing {len(logs)} logs in dataset {dataset}...')
    outputs = [None for _ in range(len(logs))]
    cache_pairs = {}

    log_chunk = []
    log_chunk_index = []

    t1 = time.time()

    t_caching = 0

    for index, log in enumerate(tqdm(logs)):

        if index % 10000 == 0 and len(cache_pairs) != 0:
            cache_pairs = dict(sorted(cache_pairs.items(), key=lambda item: item[1][1], reverse=True))
            # keys_to_remove = [k for k,v in cache_pairs.items() if v[1] == 0]
            # for k in keys_to_remove:
            #     del cache_pairs[k]

        for template, value_f in cache_pairs.items():
            match_result = matches_template(log, [value_f[0], template])
            if match_result != None and match_result in cache_pairs:
                cache_pairs[match_result][1] += 1
                outputs[index] = match_result
                break
    
        if outputs[index] == None:
            log_chunk.append(log)
            log_chunk_index.append(index)

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
                # print(f"=" * 40)
                # print(f"parsing the cluster {index} in {cluster_nums} clusters\nsample log: {c.logs[0]}")
                tmp, template, c, new_cluster, cached = parser.get_responce( c, cluster_nums, cache_pairs, [], shot)

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
    t2 = time.time()
    print(f'initial time: {t1 - t0}')
    print(f'parsing time: {t2 - t1 - t_caching}')
    print(f'caching time: {t_caching}')
    print(f'idetified templates: {len(set(outputs))}')
    # write to file
    df_new = pd.DataFrame()
    df_new['Content'] = logs
    df_new['EventTemplate'] = outputs
    df_new.to_csv(output_dir + f'{dataset}_2k.log_structured.csv', index=False)
    with open(output_dir + f'{dataset}_2k.log_templates.txt', 'w') as f:
        for template, value_f in cache_pairs.items():
            f.write("%7d:%s\n" % (value_f[1], template))
    # evaluate_single_dataset(output_dir + f'{dataset}_2k.log_structured.csv', dataset)


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
    # datasets = ['BGL', 'HDFS', 'HealthApp', 'OpenStack', 'OpenSSH', 'HPC', 'Zookeeper',
    #             'Mac', 'Hadoop', 'Android', 'Windows', 'Apache', 'Thunderbird', 'Spark', 'Linux']
    dataset_format = {
        'BGL':'<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>',
        'HDFS':'<Date> <Time> <Pid> <Level> <Component>: <Content>',
        'OpenStack':'<Timestamp> <Node> <Component> <Level> <Content>',
        'Zookeeper':'<Date> <Time> <Level> \[<Node>:<Component>@<Id>\] - <Content>',
        'OpenSSH':'<Date> <Time> <Pid> <Level> <Component> <Content>',
        }
    datasets = ['OpenSSH']
    model = args.model
    
    theme = f"LogBatcher_{args.shot}shot_{args.candidate}candidate_{args.batch_size}batchsize_{args.chunk_size}chunksize_full"

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
            log_format = dataset_format[dataset],
            output_dir=output_dir, 
            parser=parser, 
            shot=args.shot,
            candidate=args.candidate,
            batch_size=args.batch_size,
            chunk_size=args.chunk_size,
            Concurrent=False,
            sample_method = args.sample_method
            
        )
        print('time cost by llm: ', parser.time_consumption_llm)
