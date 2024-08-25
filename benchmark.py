import argparse
import json
import os
import pandas as pd
from logbatcher.parser import Parser
from logbatcher.util import generate_logformat_regex, log_to_dataframe
from LogBatcher.logbatcher.parsing_base import single_dataset_paring

def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', type=str, default='2k', choices=['2k', 'full'],
                        help='evaluate on 2k or full dataset.')
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo-0125',
                        help='the Large Lauguage model used in LogBatcher, default to be gpt-3.5-turbo-0125.')
    parser.add_argument('--batch_size', type=int, default=10, 
                        help='The size of a batch.')
    parser.add_argument('--sample_method', type=str, default='dpp', choices=['dpp', 'random', 'similar'],
                        help='Sample method: dpp, random, similar.')
    parser.add_argument('--chunk_size', type=int, default=2000,
                        help='Size of logs in a chunk.')
    parser.add_argument('--dataset', type=str, default='null')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = set_args()

    datasets = ['BGL', 'HDFS', 'OpenStack', 'OpenSSH', 'HPC', 'Zookeeper', 'Spark', 'Proxifier', 'HealthApp', 'Mac', 'Hadoop', 'Apache', 'Linux', 'Thunderbird', 'Windows', 'Android']

    # loghub-2.0 does not have dataset Windows and Android
    if args.data_type == 'full':
        datasets = datasets[:-2]

    # evaluate on a single dataset or your own dataset
    if args.dataset != 'null':
        datasets = [args.dataset]
    
    # the file name of the output
    theme = f"logbatcher_{args.data_type}"
    output_dir = f'outputs/parser/{theme}/'
    

    # load api key and dataset format
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    parser = Parser(args.model, theme, config)
    for index, dataset in enumerate(datasets):
        if os.path.exists(f'{output_dir}{dataset}_full.log_structured.csv'):
            print(f'{dataset} has been parsed, skip it.')
            continue

        # Initializing
        if args.data_type == '2k':
            structured_log_file = f'datasets/loghub-2k/{dataset}/{dataset}_2k.log_structured_corrected.csv'
        elif args.data_type == 'full':
            structured_log_file = f'datasets/loghub-2.0/{dataset}/{dataset}_full.log_structured.csv'
        else:
            raise ValueError('data_type should be 2k or full')
        
        log_file_format = 'structured'
        if log_file_format == 'structured':
            df = pd.read_csv(structured_log_file)
            logs = df['Content'].tolist()
        elif log_file_format == 'raw':
            log_file = f'dataset/{dataset}/{dataset}.log'
            with open(log_file, 'r') as f:
                log_raws = f.readlines()
            headers, regex = generate_logformat_regex(config['datasets_format'][dataset])
            logs = log_to_dataframe(log_file, regex, headers, len(log_raws))
        else:
            raise ValueError('log_file_format should be structured or raw')

        single_dataset_paring(
            dataset=dataset,
            contents=logs,
            output_dir=output_dir, 
            parser=parser, 
            batch_size=args.batch_size,
            chunk_size=args.chunk_size,
            sample_method = args.sample_method,
            data_type = args.data_type
        )
        print('time cost by llm: ', parser.time_consumption_llm)
