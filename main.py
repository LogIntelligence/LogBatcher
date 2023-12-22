import csv
import random
import time
import pandas as pd
from llm import chatGPT
from LS_to_LT import LS_to_LT_Windows
import json

# 选择数据集
datasets = ['BGL', 'HDFS', 'Linux', 'HealthApp', 'OpenStack', 'OpenSSH', 'Proxifier', 'HPC', 'Zookeeper', 'Mac',
            'Hadoop', 'Android', 'Windows', 'Apache', 'Thunderbird', 'Spark']
dataset = datasets[12]

# 读取log messages
with open('dataset\\' + dataset + '\\' + dataset + '_2k.log', 'r') as file:
    log_messages = file.readlines()

# 读取log templates
log_templates = pd.read_csv('dataset\\' + dataset + '\\' + dataset +
                 '_2k.log_structured_corrected.csv')['EventTemplate']

# 读取log statements
with open('output\\' + dataset + '.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    row_count = sum(1 for row in reader)
outputed_log_statements_nums = row_count - 1

index = outputed_log_statements_nums + 1

# 加载chatGPT chat实体类
chat = chatGPT.Chat(dataset)

# 切片并遍历
for LM,LT in zip(log_messages[outputed_log_statements_nums:],log_templates[outputed_log_statements_nums:]):
    while True:
        LS = chat.reverse_test(log_message=LM,VERBOSE=False,shot = 2)
        if LS == 'IQ' or LS == 'RL2':
            print(LS)
            if(not chat.change_api_key()):
                print('No more api key available')
                exit()
        elif LS == 'RL1':
            print(LS)
            time.sleep(20)
            continue
        else:
            LS = LS.choices[0].message.content
            LTfromLS = LS_to_LT_Windows(LS)
            break
    with open('output\\' + dataset + '.csv', mode='a', newline='') as file:
        # 写入一行新的数据
        writer = csv.writer(file)
        writer.writerow([LM, LS, LT, LTfromLS])
    print(str(index)+':'+LTfromLS)
    index += 1

with open('output\\' + dataset + '.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    row_count = sum(1 for row in reader)
outputed_log_statements_nums = row_count - 1
print('len(log_statements) = ' + str(outputed_log_statements_nums))
