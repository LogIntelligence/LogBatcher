from utils.cluster import Cluster
from utils.sample_byword import extract_variables


def prune_from_cluster(template, cluster, cluster_nums):
    new_logs = []
    new_indexs = []
    for log, index in zip(cluster.static_logs, cluster.indexs):
        if extract_variables(log, template) == []:
            new_logs.append(log)
            new_indexs.append(index)
            cluster.indexs.remove(index)
    if new_logs == []:
        return cluster, None
    else:
        print(f"prune {len(new_logs)} logs from {len(cluster.static_logs)} logs")
        return cluster, Cluster(cluster_nums, new_logs, new_indexs, '')