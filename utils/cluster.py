import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
from utils.sample_byword import dpp_sample
import random

class Cluster:
    def __init__(self, label, logs, indexs, oracle_template, remove_duplicate=False, remain_num=10, sample_method="dpp"):
        self.label = label
        self.logs = logs
        self.indexs = indexs
        self.oracle_template = oracle_template
        self.sample_method = sample_method
        self.shuffle()
        if remove_duplicate:
            self.remove_duplicate()
            if len(self.logs) > remain_num:
                self.sample(remain_num)
    
    def remove_duplicate(self):
        self.logs = list(set(self.logs))

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

def tokenize(log_content, tokenize_pattern=r'[ ,|]'):
    words = re.split(tokenize_pattern, log_content)
    new_words = []
    list = ['/', 'kb', 'sec', 'byte', 'mb']
    for index, word in enumerate(words):
        if '=' in word:
            new_words.append(word.split('=')[0])
        elif re.search(r'\d', word):
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
    vectorizer = TfidfVectorizer(tokenizer=lambda x: x, lowercase=False)
    return vectorizer.fit_transform(tokenized_logs)


def cluster(vectorized_logs):
    cluster = DBSCAN(eps=0.1, min_samples=5)
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