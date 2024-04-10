import csv
import time
import pandas as pd
from tqdm import tqdm
from llm import chatGPT
from LS_to_LT import LS_to_LT_Windows, LS_to_LT_Android


# choose dataset
datasets = ['BGL', 'HDFS', 'Linux', 'HealthApp', 'OpenStack', 'OpenSSH', 'Proxifier', 'HPC', 'Zookeeper', 'Mac',
            'Hadoop', 'Android', 'Windows', 'Apache', 'Thunderbird', 'Spark']
dataset = datasets[12]

def reverse(dataset, shot):

    # read log messages
    with open('dataset/' + dataset +
            '/' + dataset + '_2k.log', 'r') as logfile:
        log_messages = logfile.readlines()

    # red log templates(ground truth)
    log_templates = pd.read_csv('dataset/' + dataset + '/' + dataset +'_2k.log_structured_corrected.csv')['EventTemplate']

    outputfile = 'output/results/' + dataset + '_' + str(shot) + 'shot.csv'
    # read logging statements
    columns = ['LogMessage', 'LoggingStatement', 'LogTemplate', 'LogTemplate_fromLS']
    try:
        # 尝试以读模式打开文件
        df = pd.read_csv(outputfile)
    except FileNotFoundError:
        # 如果文件不存在，就新建一个DataFrame并指定列名
        df = pd.DataFrame(columns=columns)
        # 以写模式打开文件
        df.to_csv(outputfile, index=False)
    line_nums = len(df)

    if line_nums >= 2000:
        print(dataset + ': has been reversed')
        return

    if len(log_messages) != len(log_templates) or len(log_messages) != 2000:
        print(dataset + ': data nums error')
        return

    # load chatGPT chat
    chat = chatGPT.Chat(dataset)

    # 切片并遍历
    for LM, LT in tqdm(zip(log_messages[line_nums:], log_templates[line_nums:])):
        while True:
            LS = chat.reverse_test(log_message=LM, shot = shot)
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
        with open(outputfile, mode='a', newline='') as file:
            # 写入一行新的数据
            writer = csv.writer(file)
            LS_fromtemplate = LS_to_LT_Android(LS)
            writer.writerow([LM, LS, LT, LS_fromtemplate])

    df = pd.read_csv(outputfile)
    if len(df) == 2000:
        print(dataset + ' on ' + shot + 'shot reserved successfully !')