import argparse
from collections import Counter
import json
import os
import string
import pandas as pd
from utils.cluster import Cluster,tokenize, vectorize, cluster, reassign_clusters
from utils.parser import Cluster_Parser
from utils.evaluator import evaluate, evaluate_all_datasets
from utils.sample import sample_from_clusters
from utils.add_cluster import hierichical_clustering, meanshift_clustering
import pdb


def verify_template(template):
    template = template.replace("<*>", "")
    template = template.replace(" ", "")
    return any(char not in string.punctuation for char in template)

def single_dataset_paring(dataset, output_dir, parser, shot, candidate, batch_size, sample_method = 'dpp', debug = True):
    print(f'Parsing {dataset}...')

    # initialize
    df = pd.read_csv(f'dataset/{dataset}/{dataset}_2k.log_structured_corrected.csv')
    logs = df['Content'].tolist()

    # Partitioning:
    # tokenize -> vectorize -> cluster -> reassign_clusters
    tokenized_logs = [tokenize(log) for log in logs]
    labels, cluster_nums = cluster(vectorize(tokenized_logs))
    labels, cluster_nums = reassign_clusters(labels, cluster_nums, tokenized_logs)

    # another clustering
    # labels, cluster_nums = hierichical_clustering(logs, granularity="coarse")
    # labels, cluster_nums = meanshift_clustering(logs)

    # inputs, outputs and cache
    clusters = [None for _ in range(cluster_nums)]
    outputs = [None for _ in range(len(logs))]
    cache_pairs = {}
    identified_templates_num = 0

    # create clusters
    for index, label in enumerate(labels):
        if clusters[label] is None:
            clusters[label] = Cluster()
        clusters[label].append_log(logs[index], index)

    # sorting
    clusters = sorted(clusters, key=lambda cluster: len(cluster.logs), reverse=True)

    # batching
    [cluster.batching(batch_size, sample_method) for cluster in clusters]
    
    # sampling labeled data if needed
    sample_pairs = []
    # sample_pairs = sample_from_clusters(clusters, candidate)

    # Parsing
    for index, old_cluster in enumerate(clusters):

        template, old_cluster, new_cluster = parser.get_responce(old_cluster, cache_pairs, sample_pairs, shot)

        if debug:
            print('=' * 20)
            print(
                f'New cluster processed, {identified_templates_num + 1} templates identified till now:')
            print(f'Refer Log: {old_cluster.logs[0]}')
            print(f'Output Template: {template}')

        # update clusters
        if new_cluster.size != 0:
            new_cluster.batching(batch_size, sample_method)
            clusters.append(new_cluster)
            cluster_nums += 1

        # update cache
        if template not in cache_pairs and verify_template(template):
            cache_pairs[template] = [old_cluster.logs[0], 0]
            identified_templates_num += 1

        for index in old_cluster.indexs:
            outputs[index] = template
    
    # output logs
    output_log_file = output_dir + f'{dataset}_2k.log_structured.csv'
    df = pd.DataFrame({'Content': logs, 'EventTemplate': outputs})
    df.to_csv(output_log_file, index=False)

    # output templates
    counter = Counter(outputs)
    items = list(counter.items())
    items.sort(key=lambda x: x[1], reverse=True)
    output_template_file = output_dir + \
        f'{dataset}_2k.template_structured.csv'
    template_df = pd.DataFrame(items, columns=['EventTemplate', 'Occurrence'])
    template_df['EventID'] = [f"E{i + 1}" for i in range(len(template_df))]
    template_df[['EventID', 'EventTemplate', 'Occurrence']].to_csv(
        output_template_file, index=False)

    evaluate(output_dir + f'{dataset}_2k.log_structured.csv',f'dataset/{dataset}/{dataset}_2k.log_structured_corrected.csv', dataset, mismatch= False)


def set_args():
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
    parser.add_argument('--dataset', type=str, default='null')
    parser.add_argument('--time', type=int, default=1)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = set_args()
    datasets = ['BGL', 'HDFS', 'HealthApp', 'OpenStack', 'OpenSSH', 'HPC', 'Zookeeper','Mac', 'Hadoop', 'Android', 'Windows', 'Apache', 'Thunderbird', 'Spark', 'Linux', 'Proxifier']
    if args.dataset != 'null':
        datasets = [args.dataset]

    theme = f"LogBatcher_2k_{args.shot}shot_{args.candidate}candidate_{args.batch_size}batchsize_{args.model.replace('/','_')}_{args.sample_method}"
    output_dir = f'outputs/parser/{theme}/'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # load api key
    with open('config.json', 'r') as f:
        config = json.load(f)

    parser = Cluster_Parser(args.model, theme, config)
    for index, dataset in enumerate(datasets):
        if os.path.exists(f'{output_dir}{dataset}_2k.log_structured.csv'):
            print(f'{dataset} has been parsed, skip it.')
            continue
        single_dataset_paring(
            dataset=dataset,
            output_dir=output_dir,
            parser=parser,
            shot=args.shot,
            candidate=args.candidate,
            batch_size=args.batch_size,
            sample_method=args.sample_method
        )
    # evaluate_all_datasets(theme, datasets=datasets, data_tpye='2k')
