from collections import OrderedDict
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
from utils.algorithms import dpp_sample
from utils.util import mutate
import random

class Cluster:
    def __init__(self, label, logs, indexs, oracle_template, remove_duplicate=False, remain_num=10, sample_method="dpp"):
        self.label = label
        self.logs = logs
        self.indexs = indexs
        self.oracle_template = oracle_template
        self.sample_method = sample_method
        # self.shuffle()
        # self.mutation()
        if remove_duplicate:
            self.remove_duplicate()
            if len(self.logs) > remain_num:
                self.sample(remain_num)
    
    def mutation(self):
        length = len(self.logs)
        pattern1 = r'^[a-zA-Z]+[0-9]+$' # may not be a varaible
        pattern2 = r"[^012][3-9]"  # may be a varaible
        if length > 1:
            for token in tokenize(self.logs[0], removeDight=False):
                if re.search(pattern2, token) and not re.match(pattern1, token):
                    self.logs[0].replace(token, mutate(token))
    
    def remove_duplicate(self):
        # self.logs = list(set(self.logs))
        self.logs = list(OrderedDict.fromkeys(self.logs))

    def shuffle(self):
        seed = 0
        random.seed(seed)
        random.shuffle(self.logs)

    def sample(self, remain_num):
        # vetorize logs
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(self.logs)  # logs 是你的文本日志列表
        tfidf_matrix = tfidf_matrix.toarray()

        # sample
        if self.sample_method == "dpp":
            similarity_matrix = cosine_similarity(tfidf_matrix)
            result = dpp_sample(similarity_matrix, remain_num)
        elif self.sample_method == "random":
            random.seed(0)
            result = random.sample(range(0, len(self.logs)), remain_num)
        self.logs = [self.logs[i] for i in result]
        return

def tokenize(log_content, tokenize_pattern=r'[ ,|]', removeDight=True):
    words = re.split(tokenize_pattern, log_content)
    new_words = []
    list = ['/', 'kb', 'sec', 'byte', 'mb']
    list = ['/']
    for index, word in enumerate(words):
        if '=' in word:
            ws = word.split('=')
            if len(ws) <= 2:
                new_words.append(ws[0])
            else:
                # might be some parameters of a URL 
                pass 
            # new_words.append(word.split('=')[0])

        elif removeDight and re.search(r'\d', word):
            pass
        elif any(i in word.lower() for i in list):
            pass
        else:
            new_words.append(word)
    new_words = [word for word in new_words if word]   # remove null
    if new_words == []:
        new_words.append(re.sub(r'\d+(\.\d+)?', '0', log_content))
    return new_words


def vectorize(tokenized_logs):
    vectorizer = TfidfVectorizer(tokenizer=lambda x: x, lowercase=False, token_pattern=None)
    return vectorizer.fit_transform(tokenized_logs)


def cluster(vectorized_logs, eps=0.1):
    cluster = DBSCAN(eps=eps, min_samples=5)
    cluster.fit(vectorized_logs)
    labels = cluster.labels_
    cluster_nums = max(labels) + 1
    return labels, cluster_nums
    

def reassign_clusters(labels, cluster_nums, tokenized_logs):
    mergerd_logs = []
    for tokenized_log in tokenized_logs:
        mergerd_logs.append(' '.join(tokenized_log))

    for i in range(len(labels)):
        if labels[i] == -1:
            for j in range(i+1, len(labels)):
                if labels[j] == -1 and mergerd_logs[i] == mergerd_logs[j]:
                    labels[j] = cluster_nums
            labels[i] = cluster_nums
            cluster_nums += 1
    return labels, cluster_nums