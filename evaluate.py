import os
from utils.evaluator import evaluate
import pandas as pd
from IPython.display import HTML
from utils.email import Email_send


def calculate_avg(numbers):
    avg = sum(numbers) / len(numbers)
    numbers.append(avg)
    numbers = [round(num, 3) for num in numbers]
    return numbers


def evaluate_all_datasets(file_name, send_email=False):

    table_order = 'HDFS Hadoop Spark Zookeeper BGL HPC Thunderbird Windows Linux Android HealthApp Apache OpenSSH OpenStack Mac'
    datasets = table_order.split(' ')

    table_data = {
        'dataset': [],
        'GA': [],
        'PA': [],
        'ED': []
    }

    result_table_path = f'outputs/parser/result_table_{file_name}.csv'
    if os.path.exists(result_table_path):
        df = pd.read_csv(result_table_path)
    else:
        ga, pa, ed = [], [], []
        for dataset in datasets:
            table_data['dataset'].append(dataset)
            output_file = f'outputs/parser/{file_name}/{dataset}_2k.log_structured.csv'

            a, b, c, d = evaluate(output_file=output_file, groundtruth_file=f'dataset/{dataset}/{dataset}_2k.log_structured_corrected.csv', dataset=dataset)
            ga.append(a)
            pa.append(b)
            ed.append(c)

        table_data['dataset'].append('avg')
        table_data['GA'] = calculate_avg(ga)
        table_data['PA'] = calculate_avg(pa)
        table_data['ED'] = calculate_avg(ed)

        df = pd.DataFrame(table_data)
        df.to_csv(result_table_path, index=False)

    table = df.to_html(index=False)
    if send_email:
        sender = Email_send(file_name)
        sender.send_table(table)
    return table


def evaluate_single_dataset(output_file, dataset):
    evaluate(output_file, f'dataset/{dataset}/{dataset}_2k.log_structured_corrected.csv', dataset)

