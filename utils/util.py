import re
import tiktoken


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