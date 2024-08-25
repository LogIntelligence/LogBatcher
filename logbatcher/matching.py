import re
from logbatcher.cluster import Cluster

import signal

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException()

def safe_search(pattern, string, timeout=0.5):
    # 设置超时信号
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)
    try:
        result = re.search(pattern, string)
    except TimeoutException:
        result = None
    finally:
        signal.alarm(0)  # 取消超时
    return result


# @timeout(10)
def extract_variables(log, template):
    log = re.sub(r'\s+', ' ', log.strip()) # DS
    pattern_parts = template.split("<*>")
    pattern_parts_escaped = [re.escape(part) for part in pattern_parts]
    regex_pattern = "(.*?)".join(pattern_parts_escaped)
    regex = "^" + regex_pattern + "$"  
    # matches = re.search(regex, log)
    matches = safe_search(regex, log, 1)
    if matches:
        return matches.groups()
    else:
        return None

def matches_template(log, cached_pair):

    reference_log = cached_pair[0]
    template = cached_pair[1]

    # length matters
    if abs(len(log.split()) - len(reference_log.split())) > 1:
        return None

    try:
        groups = extract_variables(log, template)
    except:
        groups = None
    if groups == None:
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



def prune_from_cluster(template, cluster):

    new_cluster = Cluster()
    logs, indexs = cluster.logs, cluster.indexs
    for log, index in zip(logs, indexs):
        if extract_variables(log, template) == None:
            new_cluster.append_log(log, index)

    if new_cluster.size == 0:
        return cluster, new_cluster
    else:
        old_logs = [log for log in logs if log not in new_cluster.logs]
        old_indexs = [index for index in indexs if index not in new_cluster.indexs]
        cluster.logs = old_logs
        cluster.indexs = old_indexs
        # print(f"prune {new_cluster.size} logs from {len(logs)} logs in mathcing process")
        return cluster, new_cluster