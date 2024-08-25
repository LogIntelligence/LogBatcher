from collections import OrderedDict
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
from logbatcher.sample import group_samples_clustering, dpp_sample
import random

class Cluster:
    def __init__(self):
        self.logs = []
        self.batch_logs = []
        self.indexs = []
        self.size = 0
        

    def append_log(self, log, index):
        self.logs.append(log)
        self.indexs.append(index)
        self.size += 1
    
    def batching(self, batch_size=10, sample_method="dpp"):
        self.batch_logs = list(OrderedDict.fromkeys(self.logs)) # remove duplicates
        if len(self.batch_logs) > batch_size:
            self.sample(batch_size, sample_method)

    def sample(self, batch_size, sample_method):
        # vetorize logs
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(self.batch_logs)
        tfidf_matrix = tfidf_matrix.toarray()

        # sample
        if sample_method == "dpp":
            similarity_matrix = cosine_similarity(tfidf_matrix)
            result = dpp_sample(similarity_matrix, batch_size)
        elif sample_method == "random":
            random.seed(0)
            result = random.sample(range(0, len(self.batch_logs)), batch_size)
        elif sample_method == "similar":
            result = group_samples_clustering(tfidf_matrix, batch_size)[0]
        else:
            raise ValueError("Invalid sample method")
        self.batch_logs = [self.batch_logs[i] for i in result]
        return

def tokenize(log_content, tokenize_pattern=r'[ ,|]', removeDight=True):
    words = re.split(tokenize_pattern, log_content)
    new_words = []
    for word in words:
        if '=' in word:
            ws = word.split('=')
            if len(ws) <= 2:
                new_words.append(ws[0])
            else:
                # might be some parameters of a URL 
                pass 

        elif removeDight and re.search(r'\d', word):
            pass
        elif '/' in word.lower():
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