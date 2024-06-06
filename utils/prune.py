import sys
from utils.cluster import Cluster
from utils.matching import extract_variables

def prune_from_cluster(template, cluster, cluster_nums):
    new_logs = []
    new_indexs = []
    logs, indexs = cluster.static_logs, cluster.indexs
    for log, index in zip(logs, indexs):
        if extract_variables(log, template) == None:
            new_logs.append(log)
            new_indexs.append(index)
    if new_logs == []:
        return cluster, None
    else:
        old_logs = [log for log in logs if log not in new_logs]
        old_indexs = [index for index in indexs if index not in new_indexs]
        cluster.static_logs = old_logs
        cluster.indexs = old_indexs
        # print(f"prune {len(new_logs)} logs from {length} logs in mathcing process")
        return cluster, Cluster(cluster_nums, new_logs, new_indexs, '')
