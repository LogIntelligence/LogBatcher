import csv
import time
import pandas as pd
from tqdm import tqdm
from llm import chatGPT
from LS_to_LT import LS_to_LT_Windows

# choose dataset
datasets = ['BGL', 'HDFS', 'Linux', 'HealthApp', 'OpenStack', 'OpenSSH', 'Proxifier', 'HPC', 'Zookeeper', 'Mac',
            'Hadoop', 'Android', 'Windows', 'Apache', 'Thunderbird', 'Spark']
dataset = datasets[12]

# 
def reverse(dataset):

    # read log messages
    with open('dataset\\' + dataset +
            '\\' + dataset + '_2k.log', 'r') as logfile:
        log_messages = logfile.readlines()

    # red log templates(ground truth)
    log_templates = pd.read_csv('dataset\\' + dataset + '\\' + dataset +'_2k.log_structured_corrected.csv')['EventTemplate']

    # read nums of outputed log statements
    with open('output\\results\\' + dataset + '.csv', 'r', 'r') as outputfile:
        line_nums = len(outputfile.readlines()) - 1

    if line_nums >= 2000 or  len():
        print(dataset + ' has been reversed')
        return

    # load chatGPT chat
    chat = chatGPT.Chat(dataset)

    # 切片并遍历
    for LM, LT in tqdm(zip(log_messages[line_nums:], log_templates[line_nums:])):
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
                break
        with open('output\\' + dataset + '.csv', mode='a', newline='') as file:
            # 写入一行新的数据
            writer = csv.writer(file)
            writer.writerow([LM, LS, LT])
        index += 1

    with open('output\\' + dataset + '.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        row_count = sum(1 for row in reader)
    outputed_log_statements_nums = row_count - 1
    print('len(log_statements) = ' + str(outputed_log_statements_nums))