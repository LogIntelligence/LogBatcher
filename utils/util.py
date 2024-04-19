from collections import Counter
import random

def choose(list : list):
    """
    choose the most frequent template from the list

    Args: 
        list of templates
    Returns:
        final_template: the most frequent template
        freq: the frequency of each template
    """
    # majority vote
    freq = Counter(list)
    length = len(freq) 
    candidates = freq.most_common(len(freq))
    final_template = ''
    if length == 0:
        pass
    elif length == 1:
        final_template = candidates[0][0]
    elif not all(any (char.isdigit() for char in log) for log in list):
            list = [log for log in list if not any(char.isdigit() for char in log)]
            freq1 = Counter(list)
            candidates = freq1.most_common(len(freq1))
            final_template = candidates[0][0]
    else:
        count1 = 0
        count2 = 0
        for char in candidates[0][0]:
            if char.isdigit():
                count1 += 1
        for char in candidates[1][0]:
            if char.isdigit():
                count2 += 1
        if count1 < count2:
            final_template = candidates[1][0]
    return final_template, freq


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
