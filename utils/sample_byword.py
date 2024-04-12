import random
import pandas as pd
import re
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def dpp_sample(S, k):
    # S: similarity matrix
    # k: number of items to sample
    n = S.shape[0]

    # Initialize empty set Y
    Y = set()

    for _ in range(k):
        best_i = -1
        best_p = -1

        for i in range(n):
            if i not in Y:
                # Compute determinant of submatrix
                det_Yi = np.linalg.det(S[np.ix_(list(Y) + [i], list(Y) + [i])])

                # Compute probability of adding i to Y
                p_add = det_Yi / (1 + det_Yi)

                if p_add > best_p:
                    best_p = p_add
                    best_i = i

        # Add best item to Y
        Y.add(best_i)

    return list(Y)


def extract_variables(log, template):
    # 将模板中的 <*> 替换为正则表达式的捕获组 (.*?)
    # 为了避免正则表达式的特殊字符导致的问题，先将模板中除了 <*> 外的其他部分进行转义
    # 然后将 <*> 替换为正则表达式的捕获组
    # 这里假设模板中的 <*> 不紧邻正则特殊字符，如果有，需要更复杂的处理
    pattern_parts = template.split("<*>")
    pattern_parts_escaped = [re.escape(part) for part in pattern_parts]
    regex_pattern = "(.*?)".join(pattern_parts_escaped)
    regex = "^" + regex_pattern + "$"  # 添加开始和结束锚点以确保完整匹配

    matches = re.search(regex, log)
    if matches:
        return matches.groups()
    else:
        return []

def sample_byword(df, k, method='dpp', showLogs=False):

    logs = df['Content'].tolist()
    templates = df['EventTemplate'].tolist()
    

    # vetorize logs
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(logs)  # logs 是你的文本日志列表
    tfidf_matrix = tfidf_matrix.toarray()

    # sample
    if method == "dpp":
        similarity_matrix = cosine_similarity(tfidf_matrix)
        result = dpp_sample(similarity_matrix, 5)
    elif method == "random":
        random.seed(0)
        result = random.sample(range(0, 2000), 5)

    # extract variables
    vars = []
    for i in result:
        if showLogs:
            print(logs[i])
        variables = extract_variables(logs[i], templates[i])
        for var in variables:
            if var not in vars:
                vars.append(var)
    return set(vars)

