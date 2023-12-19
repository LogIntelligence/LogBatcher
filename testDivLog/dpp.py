import numpy as np
import math
import time


def dpp(kernel_matrix, max_length, epsilon=1E-10):
    item_size = kernel_matrix.shape[0]
    cis = np.zeros((max_length, item_size))
    di2s = np.copy(np.diag(kernel_matrix))
    selected_items = list()
    selected_item = np.argmax(di2s)
    selected_items.append(selected_item)
    while len(selected_items) < max_length:
        k = len(selected_items) - 1
        ci_optimal = cis[:k, selected_item]
        di_optimal = math.sqrt(di2s[selected_item])
        elements = kernel_matrix[selected_item, :]
        eis = (elements - np.dot(ci_optimal, cis[:k, :])) / di_optimal
        cis[k, :] = eis
        di2s -= np.square(eis)
        selected_item = np.argmax(di2s)
        # if di2s[selected_item] < epsilon:
        #     break
        selected_items.append(selected_item)
    return selected_items

def getDppIndex(log_emb_list, 
                item_size,    # log dataset size
                split_ratio):

    max_length = int(item_size * split_ratio)
    feature_vectors = np.array(log_emb_list) 

    # standarization no need for log embeddings
    feature_vectors /= np.linalg.norm(feature_vectors, axis=1, keepdims=True)

    # calculate similarity matrix of log embeddings
    similarities = np.dot(feature_vectors, feature_vectors.T) 

    t = time.time()
    result = dpp(similarities, max_length)
    result.sort()
    print('DPP algorithm running time: ' + '\t' + "{0:.4e}".format(time.time() - t))
    # print(f'DPP result indexex: {result}')
    return result


def DPPsplit(log_list, groundtruth_template, candidate_idx):
    cand_logs = [log_list[idx] for idx in candidate_idx]
    cand_templates = [groundtruth_template[idx] for idx in candidate_idx]
    test_idx = []
    for i in range(len(log_list)):
      if i not in candidate_idx: test_idx.append(i)
    test_idx.sort()
    test_logs = [log_list[idx] for idx in test_idx]
    test_templates = [groundtruth_template[idx] for idx in test_idx]
    return test_logs, cand_logs, test_templates, cand_templates