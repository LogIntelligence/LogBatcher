from concurrent.futures import ThreadPoolExecutor
import json
import os
import pandas as pd
from tqdm import tqdm
from utils.evaluator import evaluate
from utils.cluster import Cluster,tokenize, vectorize, cluster, reassign_clusters
from utils.parser import Cluster_Parser
from utils.sample_byword import sample_byword

def single_dataset_paring(dataset, output_dir, config, isConcurrent=True):
    print(f'Parsing {dataset}...')

    # initialize
    df = pd.read_csv(
        f'dataset/{dataset}/{dataset}_2k.log_structured_corrected.csv')
    logs = df['Content'].tolist()

    # get the words
    word_set = sample_byword(df, 5, method='dpp')
    config['additional_incontext'] = f"\nvaraibles:\n`{word_set}`"
    parser = Cluster_Parser(config)

    # tokenize -> vectorize -> cluster -> reassign_clusters
    tokenized_logs = [tokenize(log) for log in logs]
    labels, cluster_nums = cluster(vectorize(tokenized_logs))
    labels, cluster_nums = reassign_clusters(
        labels, cluster_nums, tokenized_logs)

    # output file
    os.makedirs(output_dir, exist_ok=True)
    f = open(output_dir + f'{dataset}.txt', 'w')

    outputs = [None for _ in range(2000)]

    inputs = []
    for i in range(cluster_nums):
        inputs.append([-1, [], [], ''])  # label, logs, indexs, ground_truth
    for i, label in enumerate(labels):
        inputs[label][0] = label
        inputs[label][1].append(logs[i])
        inputs[label][2].append(i)
        if inputs[label][3] == '':
            inputs[label][3] = df['EventTemplate'][i]

    clusters = []
    for input in inputs:
        c = Cluster(*input)
        clusters.append(c)

    # Concurrent or not
    if isConcurrent:
        templates = []
        with ThreadPoolExecutor(max_workers=16) as executor:
            templates = list(
                tqdm(executor.map(parser.get_responce, [f]*len(clusters), clusters),
                     total=len(clusters)))
        for label, template in enumerate(templates):
            for index in inputs[label][2]:
                outputs[index] = template
    else:
        for label in range(cluster_nums):
            template = parser.get_responce(f, inputs[label])
            for index in inputs[label][2]:
                outputs[index] = template

    # write to file
    f.close()
    df['Output'] = outputs
    df[['Content', 'EventTemplate', 'Output']].to_csv(
        output_dir + f'{dataset}.csv', index=False)
    evaluate(output_dir + f'{dataset}.csv', dataset)

if __name__ == "__main__":
    datasets = ['BGL', 'HDFS', 'Linux', 'HealthApp', 'OpenStack', 'OpenSSH', 'Proxifier', 'HPC',
                'Zookeeper', 'Mac', 'Hadoop', 'Android', 'Windows', 'Apache', 'Thunderbird', 'Spark']
    datasets = ['BGL', 'HDFS', 'Linux', 'HealthApp', 'OpenStack', 'OpenSSH',  'HPC', 'Zookeeper', 'Mac', 'Hadoop', 'Android', 'Windows', 'Apache', 'Spark']
    output_dir = 'outputs/parser/words_base/'

    with open('config.json', 'r') as f:
        config = json.load(f)
    config['instruction_for_batch_logs'] = "You will be provided with some varaibles and log messages. You should check if the giving log messages share the same template. If so, abstract variables with `{{placeholders}}` to extract the corresponding template.\nPrint the input log's template delimited by backticks."
    config['instruction_for_one_log'] = "You will be provided with some varaibles and a log message delimited by backticks. You must abstract variables with `{{placeholders}}` to extract the corresponding template.\nPrint the input log's template delimited by backticks."

    for index, dataset in enumerate(datasets):
        single_dataset_paring(dataset, output_dir, config, isConcurrent=True)
