from tqdm import tqdm
from nltk.metrics.distance import edit_distance


def calculate_edit_distance(groundtruth, parsedresult):
    edit_distance_result, normalized_ed_result, cache_dict = [], [] , {}
    iterable = zip(groundtruth['EventTemplate'].values, parsedresult['EventTemplate'].values)
    length_logs = len(groundtruth['EventTemplate'].values)
    iterable = tqdm(iterable, total=length_logs)
    for i, j in iterable:
        if i != j:
            if (i, j) in cache_dict:
                ed = cache_dict[(i, j)]
            else:
                ed = edit_distance(i, j)
                cache_dict[(i, j)] = ed
            normalized_ed = 1 - ed / max(len(i), len(j))
            edit_distance_result.append(ed)
            normalized_ed_result.append(normalized_ed)

    accuracy_ED = sum(edit_distance_result) / length_logs
    accuracy_NED = (sum(normalized_ed_result) + length_logs - len(normalized_ed_result)) / length_logs
    print('Normalized_Edit_distance (NED): %.4f, ED: %.4f,'%(accuracy_NED, accuracy_ED))
    return accuracy_ED, accuracy_NED