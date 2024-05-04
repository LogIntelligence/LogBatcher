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

    table_order = 'HDFS Hadoop Spark Zookeeper BGL HPC Thunderbird Windows Linux Android HealthApp Apache Proxifier OpenSSH OpenStack Mac'
    datasets = table_order.split(' ')

    table_data = {
        'dataset': [],
        'GA': [],
        'PA': [],
        'ED': []
    }

    result_table_path = f'outputs/parser/{file_name}/result_table.csv'
    if os.path.exists(result_table_path):
        df = pd.read_csv(result_table_path)
    else:
        ga, pa, ed = [], [], []
        for dataset in datasets:
            table_data['dataset'].append(dataset)
            file_path = f'outputs/parser/{file_name}/{dataset}.csv'

            a, b, c, d = evaluate(file_path, dataset)
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

