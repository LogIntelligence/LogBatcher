import json
import os
import time
import pandas as pd
from collections import Counter, OrderedDict
from tqdm import tqdm
from logbatcher.cluster import Cluster,tokenize, vectorize, cluster, reassign_clusters
from logbatcher.additional_cluster import hierichical_clustering,meanshift_clustering
from logbatcher.matching import matches_template
from logbatcher.util import verify_template


def single_dataset_paring(dataset, contents, output_dir, parser, batch_size = 10, chunk_size = 2000 , sample_method = 'dpp', clustering_method = 'dbscan', data_type = '2k', debug=True):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logs = contents
    identified_templates_num = 0
    cache_pairs = {}
    log_chunk = []
    log_chunk_index = []
    remove_duplicate = True if data_type == 'full' else False
    cache_sort_step = len(logs) // 100
    print(f'Parsing {len(logs)} logs in dataset {dataset}...')

    # temp to store parsing results
    if remove_duplicate:
        temp_logs = logs.copy()
        logs = list(OrderedDict.fromkeys(logs))
    outputs = [None for _ in range(len(logs))]
    
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
            if clustering_method == 'dbscan':
                # tokenize -> vectorize -> cluster -> reassign_clusters
                tokenized_logs = [tokenize(log) for log in log_chunk]
                labels, cluster_nums = cluster(vectorize(tokenized_logs))
                labels, cluster_nums = reassign_clusters(labels, cluster_nums, tokenized_logs)
            elif clustering_method == 'hierarchical':
                labels, cluster_nums = hierichical_clustering(log_chunk)
            elif clustering_method == 'meanshift':
                labels, cluster_nums = meanshift_clustering(log_chunk)
            else:
                raise ValueError('Invalid clustering method')

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
                template, old_cluster, new_cluster = parser.get_responce(old_cluster, cache_pairs, dataset = dataset, data_type = data_type)

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
    
    if remove_duplicate:
        print("map the outputs")
        log_to_index = {log: index for index, log in enumerate(logs)}
        temp_outputs = []
        for log in tqdm(temp_logs):
            temp_outputs.append(outputs[log_to_index[log]])
        outputs = temp_outputs
        logs = temp_logs
    
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
