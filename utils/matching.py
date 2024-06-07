import random
import re
from utils.cluster import Cluster

def extract_variables(log, template):
    log = re.sub(r'\s+', ' ', log.strip()) # DS
    pattern_parts = template.split("<*>")
    pattern_parts_escaped = [re.escape(part) for part in pattern_parts]
    regex_pattern = "(.*?)".join(pattern_parts_escaped)
    regex = "^" + regex_pattern + "$"  
    matches = re.search(regex, log)
    if matches:
        return matches.groups()
    else:
        return None

def matches_template(log, cached_pair):

    reference_log = cached_pair[0]
    template = cached_pair[1]

    # length matters
    if len(log.split()) != len(reference_log.split()):
        return None
    
    groups = extract_variables(log, template)

    if not groups:
        return None

    # consider the case where the varaible is empty
    parts = []
    for index, part in enumerate(template.split("<*>")):
        parts.append(part)
        if index < len(groups):
            if groups[index] == '':
                parts.append('')
            else:
                parts.append('<*>')

    return ''.join(parts)



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