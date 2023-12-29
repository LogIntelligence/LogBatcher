import pandas as pd
import re

def LS_to_LT_Spark(text):
    # 提取引号中的文本
    quoted_text = re.findall(r'"([^"]*)"', text)

    # 将形如$freeMemory的占位符更改为<*>
    for i in range(len(quoted_text)):
        quoted_text[i] = re.sub(r'\$\w+', '<*>', quoted_text[i])

        return ' '.join(quoted_text)


def LS_to_LT_Windows(code):
    # 提取引号中的内容
    pattern = r'\"(.+?)\"'
    match = re.search(pattern, code)
    if match:
        log_msg = match.group(1)
        # 将所有类似%d的占位符转化为<*>
        template = re.sub(r'%\w+', '<*>', log_msg)
        return template
    else:
        return None


def LS_to_LT_Android(code):
    # 提取引号中的内容
    pattern = r'\"(.+?)\"'
    match = re.search(pattern, code)
    if match:
        log_msg = match.group(1)
        # 将所有类似%d的占位符转化为<*>
        template = re.sub(r'%\w+', '<*>', log_msg)
        return template
    else:
        return None

def LS_to_LT_all(dataset):
    # 读取CSV文件
    df = pd.read_csv('output\\' + dataset + '.csv')
    # 对LogStatement列的每一行应用replace_placeholders函数，并将结果存储在LogTemplate_fromLS列
    df['LogTemplate_fromLS'] = df['LogStatement'].apply(LS_to_LT_Windows)
    # 将修改后的DataFrame写回CSV文件
    df.to_csv('output\\' + dataset + '.csv', index=False)


LS_to_LT_all('Windows')



