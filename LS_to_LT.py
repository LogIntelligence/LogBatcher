import json
from llm import chatGPT
import pandas as pd
import time
import random
import csv


import time
import re

def LS_to_LT(text):
    # 提取引号中的文本
    quoted_text = re.findall(r'"([^"]*)"', text)

    # 将形如$freeMemory的占位符更改为<*>
    for i in range(len(quoted_text)):
        quoted_text[i] = re.sub(r'\$\w+', '<*>', quoted_text[i])

        return ' '.join(quoted_text)

def LS_to_LT_all(dateset):
    # 读取CSV文件
    df = pd.read_csv('output\\' + "Spark" + '.csv')
    # 对LogStatement列的每一行应用replace_placeholders函数，并将结果存储在LogTemplate_fromLS列
    df['LogTemplate_fromLS'] = df['LogStatement'].apply(LS_to_LT)
    # 将修改后的DataFrame写回CSV文件
    df.to_csv('output\\' + "Spark" + '.csv', index=False)




# # 测试
# text = 'logInfo(s"Block $blockId stored as bytes in memory (estimated size $estimatedSize, free $freeMemory)")'
# print(replace_placeholders(text))





