import random
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
    # <*> -> (.*?)
    pattern_parts = template.split("<*>")
    pattern_parts_escaped = [re.escape(part) for part in pattern_parts]
    regex_pattern = "(.*?)".join(pattern_parts_escaped)
    regex = "^" + regex_pattern + "$"  

    matches = re.search(regex, log)
    if matches:
        return matches.groups()
    else:
        return []

def matches_template(log, cached_pair):

    # length matters
    if len(log.split()) != len(cached_pair[0].split()):
        return None
    
    pattern_parts = cached_pair[1].split("<*>")
    pattern_parts_escaped = [re.escape(part) for part in pattern_parts]
    regex_pattern = "(.*?)".join(pattern_parts_escaped)
    regex = "^" + regex_pattern + "$"  
    matches = re.search(regex, log)

    if not matches:
        return None  # 如果没有匹配，返回None

    # 生成新的template
    new_template_parts = []
    for index, part in enumerate(pattern_parts):
        new_template_parts.append(part)
        if index < len(matches.groups()):
            # 如果对应的匹配是空字符串，也在新模板中放置空字符串
            if matches.groups()[index] == '':
                new_template_parts.append('')
            else:
                new_template_parts.append('<*>')

    # 由于最后一个部分后面不需要加'<*>', 我们需要拼接处理后的模板部分
    new_template = ''.join(new_template_parts)
    return new_template

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

