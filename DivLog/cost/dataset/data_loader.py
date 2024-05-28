import json

from datasets import load_dataset
import pandas as pd
import os
import json


def load_train_data(r_dir=".", dataset="Apache", shot=4):
    dataset = load_dataset('json', data_files=f'{r_dir}/{dataset}/{shot}shot/1.json')
    examples = [(x['text'], x['label']) for x in dataset['train']]
    return examples


def load_test_data(r_dir=".", dataset="Apache"):
    logs = pd.read_csv(f"{r_dir}/{dataset}/{dataset}_2k.log_structured_corrected.csv")
    return logs.Content.tolist()


def sample_one_shot(r_dir="."):
    for dataset in os.listdir(r_dir):
        if os.path.isdir(os.path.join(r_dir, dataset)) and "_" not in dataset:
            df = pd.read_csv(f"{r_dir}/{dataset}/{dataset}_2k.log_structured_corrected.csv")
            row_id = df['Content'].value_counts()[0]
            log, label = df.iloc[row_id].Content, df.iloc[row_id].EventTemplate
            os.makedirs(f'{r_dir}/{dataset}/1shot', exist_ok=True)
            with open(f'{r_dir}/{dataset}/1shot/1.json', mode="w") as f:
                f.write(json.dumps({'text': log, 'label': label}))


if __name__ == '__main__':
    print(load_train_data(shot=2))
