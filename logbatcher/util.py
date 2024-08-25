import re
import string

import pandas as pd
import tiktoken

def data_loader(file_name, dataset_format, file_format):
    if file_format == 'structured':
        df = pd.read_csv(file_name)
        contents = df['Content'].tolist()
    elif file_format == 'raw':
        with open(file_name, 'r') as f:
            log_raws = f.readlines()
        headers, regex = generate_logformat_regex(dataset_format)
        contents = log_to_dataframe(file_name, regex, headers, len(log_raws))
    return contents


def count_prompt_tokens(prompt, model_name):
    """
    Count the number of tokens in the prompt
    Models supported: gpt-4, gpt-3.5-turbo
    """
    if model_name == "gpt-4":
        encoder = tiktoken.encoding_for_model("gpt-4")
    elif model_name == "gpt-3.5-turbo":
        encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")
    else:
        raise ValueError("未知的模型名称")

    # 计算编码后的token数
    prompt_tokens = encoder.encode(prompt)
    return len(prompt_tokens)


def count_message_tokens(messages, model_name="gpt-3.5-turbo"):
    """
    Count the number of tokens in the messages
    Models supported: gpt-4, gpt-3.5-turbo
    """
    if model_name == "gpt-4":
        encoder = tiktoken.encoding_for_model("gpt-4")
    elif model_name == "gpt-3.5-turbo":
        encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")
    else:
        raise ValueError("未知的模型名称")

    # 初始化token计数
    token_count = 0

    # 计算每个消息的token数
    for message in messages:
        role_tokens = encoder.encode(message['role'])
        content_tokens = encoder.encode(message['content'])
        token_count += len(role_tokens) + len(content_tokens) + 4  # 加上特殊的消息分隔符的token数
        # token_count +=  len(content_tokens) + 4  # 加上特殊的消息分隔符的token数
        # print(token_count)
    return token_count


def generate_logformat_regex(logformat):
        """ 
        Function to generate regular expression to split log messages
        Args:
            logformat: log format, a string
        Returns:
            headers: headers of log messages
            regex: regular expression to split log messages
        """
        headers = []
        splitters = re.split(r'(<[^<>]+>)', logformat)
        regex = ''
        for k in range(len(splitters)):
            if k % 2 == 0:
                splitter = re.sub(' +', '\\\s+', splitters[k])
                regex += splitter
            else:
                header = splitters[k].strip('<').strip('>')
                regex += '(?P<%s>.*?)' % header
                headers.append(header)
        regex = re.compile('^' + regex + '$')
        return headers, regex


def log_to_dataframe(log_file, regex, headers, size):
        """ 
        Function to transform log file to contents
        Args:
            log_file: log file path
            regex: regular expression to split log messages
            headers: headers of log messages
            size: number of log messages to read
        Returns:
            log_messages: list of log contents
        """
        log_contents = []
        with open(log_file, 'r') as file:
            for line in [next(file) for _ in range(size)]:
                try:
                    match = regex.search(line.strip())
                    message = [match.group(header) for header in headers]
                    log_contents.append(message[-1])
                except Exception as e:
                    pass
        return log_contents


def not_varibility(logs):
    a_logs = [re.sub(r'\d+', '', log) for log in logs]
    if len(set(a_logs)) == 1:
        return True
    return False

def verify_template(template):
    template = template.replace("<*>", "")
    template = template.replace(" ", "")
    return any(char not in string.punctuation for char in template)

if __name__ == "__main__":
    import json
    import csv

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