import pandas as pd
import re
import heapq
from collections import Counter, defaultdict, deque, OrderedDict
from sklearn.feature_extraction._stop_words import ENGLISH_STOP_WORDS
import time
import calendar
import random
import os
from sklearn.cluster import MeanShift
from sklearn.feature_extraction.text import TfidfVectorizer



class Vocab:
    def __init__(self, stopwords=["<*>"]):
        stopwords = [
            "a",
            "an",
            "and",
            "i",
            "ie",
            "so",
            "to",
            "the",

        ] + list(calendar.day_name) + list(calendar.day_abbr) \
          + list(calendar.month_name) + list(calendar.month_abbr)
        self.token_counter = Counter()
        self.stopwords = frozenset(set(stopwords))
        #print(self.__filter_stopwords(['LDAP', 'Built', 'with']))

    def build(self, sequences):
        print("Build vocab with examples: ", len(sequences))
        for sequence in sequences:
            sequence = self.__filter_stopwords(sequence)
            #print(sequence)
            self.update(sequence)

    def update(self, sequence):
        sequence = self.__filter_stopwords(sequence)
        self.token_counter.update(sequence)

    def topk_tokens(self, sequence, topk=3):
        sequence = self.__filter_stopwords(sequence)
        token_count = [(token, self.token_counter[token]) for token in set(sequence)]
        topk_tuples = heapq.nlargest(topk, token_count, key=lambda x: x[1])
        topk_keys = tuple([t[0] for t in topk_tuples])
        return topk_keys

    def __len__(self):
        return len(self.token_counter)

    def __filter_stopwords(self, sequence):
        return [
            token
            for token in sequence
            if (len(token) > 2) and (token not in self.stopwords)
        ]


def clean(s):
    log_format = re.sub(r'[0-9A-Za-z, ]+', '', s)
    unique_chars = list(set(log_format))
    sorted_string = ''.join(sorted(unique_chars))
    s = re.sub(':|\(|\)|=|,|"|\{|\}|@|$|\[|\]|\||;|\.?!', ' ', s)
    s = " ".join([word for word in s.strip().split() if not bool(re.search(r'\d', word))])
    # trantab = str.maketrans(dict.fromkeys(list(string.punctuation)))
    return s, sorted_string


def h_clustering(contents):
    t1 = time.time()
    vocab = Vocab()
    vocab.build([v[0].split() for v in contents.values()])
    t2 = time.time()
    # print("Build time: ", t2 - t1)

    # hierichical clustering
    hierichical_clusters = {}
    for k, v in contents.items():
        frequent_token = tuple(sorted(vocab.topk_tokens(v[0].split(), 3))) 
        log_format = v[1]
        if frequent_token not in hierichical_clusters:
            hierichical_clusters[frequent_token] = {"size": 1, "cluster": {log_format: [k]}}
        else:
            hierichical_clusters[frequent_token]["size"] = hierichical_clusters[frequent_token]["size"] + 1
            if log_format not in hierichical_clusters[frequent_token]["cluster"]:
                hierichical_clusters[frequent_token]["cluster"][log_format] = [k]
            else:
                hierichical_clusters[frequent_token]["cluster"][log_format].append(k)
    print("Number of coarse-grained clusters: ", len(hierichical_clusters.keys()))
    total_coarse_clusters = len(hierichical_clusters.keys())
    total_fine_clusters = 0
    for k, v in hierichical_clusters.items():
        total_fine_clusters += len(hierichical_clusters[k]["cluster"])
    print("Number of fine-grained clusters: ", total_fine_clusters)
    return hierichical_clusters, total_coarse_clusters, total_fine_clusters


def assign_labels(clusters, logs, granularity="coarse"):
    # Initialize the labels list with -1 for all logs
    labels = [-1] * len(logs)

    # Map each log ID to its cluster ID
    cluster_id = 0
    for frequent_tokens, cluster_info in clusters.items():
        if granularity == "coarse":
            # Assign cluster ID based on frequent tokens
            for log_format, log_ids in cluster_info["cluster"].items():
                for log_id in log_ids:
                    labels[log_id] = cluster_id
            cluster_id += 1
        elif granularity == "fine":
            # Assign unique cluster ID for each log format within frequent tokens
            for log_format, log_ids in cluster_info["cluster"].items():
                for log_id in log_ids:
                    labels[log_id] = cluster_id
                cluster_id += 1

    return labels

def hierichical_clustering(logs, granularity="coarse"):
    contents = {}
    for i, x in enumerate(logs):
        x, fx = clean(x)
        if len(x.split()) > 1:
            contents[i] = (x, fx)
    clusters, a, b = h_clustering(contents)
    labels = assign_labels(clusters, logs, granularity)
    if granularity == "coarse":
        return labels, a
    else:
        return labels, b

def replace_numbers_with_zero(text):
    return re.sub(r'\d+(\.\d+)?', '0', text)


def meanshift_clustering(logs):
    
    text_column = [replace_numbers_with_zero(log) for log in logs]

    # Text preprocessing and vectorization
    vectorizer = TfidfVectorizer()
    data_matrix = vectorizer.fit_transform(text_column).toarray()

    # Mean Shift clustering
    mean_shift = MeanShift(bandwidth=0.5)
    labels = mean_shift.fit_predict(data_matrix).tolist()
    return labels, max(labels) + 1