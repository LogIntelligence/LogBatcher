from collections import Counter
import random

def choose(tmps: list, templates : list):
    """
    choose the most frequent template from the list

    Args: 
        tmps: the list of output templates
        templates: the list of postprocess templates
    Returns:
        final_template: the most frequent template
        freq: the frequency of the postprocess templates
        freq_tmp: the frequency of the output templates
    """

    freq_tmp = Counter(tmps)
    freq = Counter(templates)
    candidates = freq.most_common(len(freq))


    # discard the template that contains digit if there exists a template that does not contain digit
    if not all(any(char.isdigit() for char in tmp) for tmp in templates):
        candidates = [candidate for candidate in candidates if not any(
            char.isdigit() for char in candidate[0])]
    
    # if there is no template, return empty string
    if len(candidates) == 0:
        final_template = ''

    # if there is only one template, return the template
    elif len(candidates) == 1:
        final_template = candidates[0][0]
        

    # check if there is a template that does not contain '<*>'
    elif not all('<*>' in candidate[0] for candidate in candidates):
        for candidate in candidates:
            if '<*>' not in candidate[0]:
                final_template = candidate[0]
                break
    
    # majority vote
    else:
        final_template = candidates[0][0]

    return final_template, freq, freq_tmp


def mutate(token : str):
    """
    randomly change the number in the token

    Args:
        token: the token to be mutated

    Returns:
        token: the mutated token
    """
    random_number = random.randint(0, 9)
    token_list = list(token)
    for index, char in enumerate(token_list):
        if char.isdigit():
            token_list[index] = str((int(char)+random_number) % 10)
    return ''.join(token_list)

def truncate(logs : list, max_length : int):
    """
    truncate the logs to a certain length

    Args:
        logs: the logs to be truncated
        max_length: the maximum length of the logs

    Returns:
        logs: the truncated logs
    """

    
    length = sum(len(log) for log in logs)
    # truncate
    # while (length > max_length):
    #     logs = logs[0::2]
    #     length = sum(len(log) for log in logs)

    prompt = '\n'.join(logs)
    return prompt
