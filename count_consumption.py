import json
import csv
from utils.util import count_message_tokens

# LogBacther
with open('/root/LogBatcher/messages.json', 'r') as file:
    messages_dict = json.load(file)
data = []
datasets = ['BGL', 'HDFS', 'OpenStack', 'OpenSSH', 'HPC', 'Zookeeper', 'Spark', 'Proxifier', 'HealthApp', 'Mac', 'Hadoop', 'Apache', 'Linux', 'Thunderbird']

all = 0

for dataset in datasets:
    messages = messages_dict[dataset]
    count = 0
    for message in messages:
        count += count_message_tokens(message)
    print(f"{dataset}: [{count}, {len(messages)}] -> {count/len(messages).__round__(2)}")
    data.append([dataset, count, len(messages), (count/len(messages)).__round__(2)])
    all += count

print(f"all: {all}")

with open('/root/LogBatcher/output_lilac_0.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Dataset", "Value1", "Value2", "Value3"])  # 写入标题
    for row in data:
        writer.writerow([row[0], row[1], row[2], row[3]])  # 写入数据