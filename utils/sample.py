import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from utils.algorithms import dpp_sample, entropy_calculate
from utils.sample_byword import extract_variables
import random
from sklearn.cluster import KMeans
import numpy as np

# entropy based sampling
# messages.append({"role": "user", "content": '2017-07-02 15:46:41.445 ksfetch[32435/0x7fff79824000] [lvl=2] main() ksfetch fetching URL (<NSMutableURLRequest: 0x1005110b0> { URL: https://tools.google.com/service/update2?cup2hreq=53f725cf03f511fab16f19e789ce64aa1eed72395fc246e9f1100748325002f4&cup2key=7:1132320327 }) to folder:/tmp/KSOutOfProcessFetcher.YH2CjY1tnx/download'})
# messages.append({"role": "assistant", "content": '`{{timestamp}} ksfetch[{{process_and_thread_id}}] [lvl={{log_level}}] main() ksfetch fetching URL (<NSMutableURLRequest: {{request_id}}> { URL: {{request_url}} }) to folder:{{folder_path}}`'})

def sample_from_clusters(clusters, shot = 32):
    clusters = sorted(clusters, key=lambda cluster: len(cluster.indexs), reverse=True)
    # form a random list
    random.seed(0)
    random_int_list = [random.randint(0, 1000) for _ in range(10)]

    sample_clusters = []
    sample_pairs = []
    for cluster in clusters:
        if len(sample_clusters) >= shot:
            break
        if cluster.oracle_template not in [pair[1] for pair in sample_clusters]:
            sample_clusters.append((cluster, cluster.oracle_template))

    for random_int in random_int_list:
        if len(sample_pairs) >= shot:
            break
        for item in sample_clusters:
            length = len(item[0].logs)
            if len(sample_pairs) >= shot:
                break
            else:
                sample_pairs.append((item[0].logs[random_int%length], item[1]))
    return sample_pairs


def nearest_k_pairs_from_log(log, sample_pairs, k):
    # Calculate similarity
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([log] + [pair[0] for pair in sample_pairs])
    similarity_matrix = cosine_similarity(tfidf_matrix)
    similarity = similarity_matrix[0][1:]
    
    # Get the nearest k pairs
    nearest_k_indices = similarity.argsort()[-k:][::-1]
    nearest_k_pairs = [sample_pairs[i] for i in nearest_k_indices]
    
    return nearest_k_pairs

def sample_based_on_entropy(dataset, shot = 5):
    # sample log-template pairs from other datasets
    datasets = ['BGL', 'HDFS', 'Linux', 'HealthApp', 'OpenStack', 'OpenSSH', 'Proxifier', 'HPC', 'Zookeeper', 'Mac',
            'Hadoop', 'Android', 'Windows', 'Apache', 'Thunderbird', 'Spark']
    datasets.remove(dataset)
    pairs =[]
    templates = []
    for d in datasets:
        df = pd.read_csv(f'dataset\{d}\{d}_2k.log_structured_corrected.csv')
        list1 = df['Content'].tolist()
        list2 = df['EventTemplate'].tolist()
        for log, template  in zip(list1, list2):
            if template not in templates:
                pairs.append((log, template, d))
                templates.append(template)

    # filter
    # for pair in pairs:
    #     if len(pair[0]) >= 500:
    #         pairs.remove(pair)
    return entropy_calculate(pairs, shot, type = 'pair')

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


def group_samples_clustering(embed_matrix, num_in_batch):
    def _calculate_cos_similarities(v1: np.array, v2: np.array):
        num = np.dot(v1, v2.T)
        denom = np.linalg.norm(v1, axis=1).reshape(-1, 1) * \
            np.linalg.norm(v2, axis=1)
        similarity_matrix = num / denom
        similarity_matrix[np.isneginf(similarity_matrix)] = 0
        similarity_matrix = 0.5 + 0.5 * similarity_matrix
        return similarity_matrix

    if embed_matrix.shape[0] % num_in_batch:
        n_clusters = embed_matrix.shape[0] // num_in_batch + 1
    else:
        n_clusters = embed_matrix.shape[0] // num_in_batch

    # K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=0,
                    n_init="auto").fit(embed_matrix)
    similarity_matrix = _calculate_cos_similarities(
        embed_matrix, kmeans.cluster_centers_)  # [n_samples, n_clusters]
    similarity_rankings = np.argsort(-similarity_matrix, axis=1)
    groups = [[] for _ in range(n_clusters)]
    for sample_idx, label in enumerate(kmeans.labels_):
        groups[label].append(sample_idx)
    # Reassign to equalize the number of samples in each cluster
    for group_idx, group in enumerate(groups):
        if len(group) > num_in_batch:
            groups[group_idx] = sorted(
                group, key=lambda x: similarity_matrix[x, group_idx], reverse=True)
            samples_to_reassign = groups[group_idx][num_in_batch:]
            groups[group_idx] = groups[group_idx][:num_in_batch]
            for sample_idx in samples_to_reassign:
                for candi_group_idx in similarity_rankings[sample_idx]:
                    if len(groups[candi_group_idx]) < num_in_batch:
                        groups[candi_group_idx].append(sample_idx)
                        break
    return groups
