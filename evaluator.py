import scipy.special
import pandas as pd
from nltk.metrics.distance import edit_distance
from sklearn.metrics import accuracy_score
import numpy as np


def evaluate(file, dataset):

    df = pd.read_csv(file)

    # Remove invalid groundtruth event Ids
    # null_logids = df[~df['EventTemplate'].isnull()].index
    # df = df.loc[null_logids]

    accuracy_exact_string_matching = accuracy_score(np.array(df.EventTemplate.values, dtype='str'),
                                                    np.array(df.Output.values, dtype='str'))
    
    # 找出不匹配的值
    # df_mismatch = df[df.EventTemplate != df.Output]
    # df_mismatch.to_csv(f'outputs/enhanced_gpt/1shot/mismatch/{dataset}.csv', index=False)


    edit_distance_result = []
    for i, j in zip(np.array(df.EventTemplate.values, dtype='str'),
                    np.array(df.Output.values, dtype='str')):
        edit_distance_result.append(edit_distance(i, j))

    edit_distance_result_mean = np.mean(edit_distance_result)
    edit_distance_result_std = np.std(edit_distance_result)

    (precision, recall, f_measure, accuracy_PA) = get_accuracy(df['EventTemplate'],
                                                               df['Output'])

    print(
        'Precision: %.4f, Recall: %.4f, F1_measure: %.4f, Group Accuracy: %.4f, Message-Level Accuracy: %.4f, Edit Distance: %.4f' % (
            precision, recall, f_measure, accuracy_PA, accuracy_exact_string_matching, edit_distance_result_mean))

    return accuracy_PA, accuracy_exact_string_matching, edit_distance_result_mean, edit_distance_result_std


def get_accuracy(series_groundtruth, series_parsedlog, debug=False):
    series_groundtruth_valuecounts = series_groundtruth.value_counts()
    real_pairs = 0
    for count in series_groundtruth_valuecounts:
        if count > 1:
            real_pairs += scipy.special.comb(count, 2)

    series_parsedlog_valuecounts = series_parsedlog.value_counts()
    parsed_pairs = 0
    for count in series_parsedlog_valuecounts:
        if count > 1:
            parsed_pairs += scipy.special.comb(count, 2)

    accurate_pairs = 0
    accurate_events = 0  # determine how many lines are correctly parsed
    for parsed_eventId in series_parsedlog_valuecounts.index:
        logIds = series_parsedlog[series_parsedlog == parsed_eventId].index
        series_groundtruth_logId_valuecounts = series_groundtruth[logIds].value_counts(
        )
        error_eventIds = (
            parsed_eventId, series_groundtruth_logId_valuecounts.index.tolist())
        error = True
        if series_groundtruth_logId_valuecounts.size == 1:
            groundtruth_eventId = series_groundtruth_logId_valuecounts.index[0]
            if logIds.size == series_groundtruth[series_groundtruth == groundtruth_eventId].size:
                accurate_events += logIds.size
                error = False
        if error and debug:
            print('(parsed_eventId, groundtruth_eventId) =',
                  error_eventIds, 'failed', logIds.size, 'messages')
        for count in series_groundtruth_logId_valuecounts:
            if count > 1:
                accurate_pairs += scipy.special.comb(count, 2)

    precision = float(accurate_pairs) / parsed_pairs
    recall = float(accurate_pairs) / real_pairs
    f_measure = 2 * precision * recall / (precision + recall)
    accuracy = float(accurate_events) / series_groundtruth.size
    return precision, recall, f_measure, accuracy