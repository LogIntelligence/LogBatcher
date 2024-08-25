import re

def post_process(response, data_type):

    response = response.replace('\n', '')
    first_backtick_index = response.find('`')
    last_backtick_index = response.rfind('`')
    if first_backtick_index == -1 or last_backtick_index == -1 or first_backtick_index == last_backtick_index:
        tmps = []
    else:
        tmps = response[first_backtick_index: last_backtick_index + 1].split('`')
    for tmp in tmps:
        if tmp.replace(' ','').replace('<*>','') == '':
            tmps.remove(tmp)
    tmp = ''
    if len(tmps) == 1:
        tmp = tmps[0]
    if len(tmps) > 1:
        tmp = max(tmps, key=len)

    template = re.sub(r'\{\{.*?\}\}', '<*>', tmp)

    if data_type == 'full':
        template = correct_single_template_full(template)
    else:
        template = correct_single_template(template)

    if template.replace('<*>', '').replace(' ','') == '':
        template = ''

    return template


def post_process_for_batch_output(response):
    outputs = response.strip('\n').split('\n')
    templates = []
    for output in outputs:
        template = re.sub(r'\{\{.*?\}\}', '<*>', output)
        template = correct_single_template(template)
        if template.replace('<*>', '').strip() == '':
            template = ''
        if template not in templates:
            templates.append(template)
    return templates

def correct_single_template(template, user_strings=None):
    """Apply all rules to process a template.

    DS (Double Space)
    BL (Boolean)
    US (User String)
    DG (Digit)
    PS (Path-like String)
    WV (Word concatenated with Variable)
    DV (Dot-separated Variables)
    CV (Consecutive Variables)

    """

    boolean = {'true', 'false'}
    default_strings = {'null', 'root', 'admin'}
    path_delimiters = {  # reduced set of delimiters for tokenizing for checking the path-like strings
        r'\s', r'\,', r'\!', r'\;', r'\:',
        r'\=', r'\|', r'\"', r'\'',
        r'\[', r'\]', r'\(', r'\)', r'\{', r'\}'
    }
    token_delimiters = path_delimiters.union({  # all delimiters for tokenizing the remaining rules
        r'\.', r'\-', r'\+', r'\@', r'\#', r'\$', r'\%', r'\&', 
    })

    if user_strings:
        default_strings = default_strings.union(user_strings)

    # apply DS
    # Note: this is not necessary while postprorcessing
    template = template.strip()
    template = re.sub(r'\s+', ' ', template)

    # apply PS
    p_tokens = re.split('(' + '|'.join(path_delimiters) + ')', template)
    new_p_tokens = []
    for p_token in p_tokens:
        # print(p_token)
        # if re.match(r'^(\/[^\/]+)+$', p_token) or re.match(r'^([a-zA-Z0-9-]+\.){2,}[a-zA-Z]+$', p_token):
        # if re.match(r'^(\/[^\/]+)+$', p_token):
        if re.match(r'^(\/[^\/]+)+\/?$', p_token) or re.match(r'^([a-z0-9-]+\.){2,}[a-z]+$', p_token):
            p_token = '<*>'
        if all(x in p_token for x in {'<*>', '.', '/'}):
            p_token = '<*>'
        new_p_tokens.append(p_token)
    template = ''.join(new_p_tokens)

    # tokenize for the remaining rules
    tokens = re.split('(' + '|'.join(token_delimiters) + ')', template)  # tokenizing while keeping delimiters
    new_tokens = []
    for token in tokens:
        # apply BL, US
        for to_replace in boolean.union(default_strings):
            if token.lower() == to_replace.lower():
                token = '<*>'

        # apply DG
        # Note: hexadecimal num also appears a lot in the logs
        # if re.match(r'^\d+$', token) or re.match(r'\b0[xX][0-9a-fA-F]+\b', token) or len(re.findall(r'\d', token)) >= 4:
        if exclude_digits(token):
            token = '<*>'

        # apply WV
        if re.match(r'^[^\s\/]*<\*>[^\s\/]*$', token):
            # if token != '<*>/<*>':  # need to check this because `/` is not a deliminator
            token = '<*>'

        # collect the result
        new_tokens.append(token)

    # make the template using new_tokens
    template = ''.join(new_tokens)

    # Substitute consecutive variables only if separated with any delimiter including "." (DV)
    while True:
        prev = template
        template = re.sub(r'<\*>\.<\*>', '<*>', template)
        if prev == template:
            break

    # Substitute consecutive variables only if not separated with any delimiter including space (CV)
    # NOTE: this should be done at the end
    while True:
        prev = template
        template = re.sub(r'<\*><\*>', '<*>', template)
        if prev == template:
            break

    while "<*>:<*>" in template:
        template = template.replace("<*>:<*>", "<*>")

    while "<*>/<*>" in template:
        template = template.replace("<*>/<*>", "<*>")

    while " -<*>" in template:
        template = template.replace(" -<*>", " <*>")

    template = re.sub(r'<\*> [KGTM]?B\b', '<*>', template)


    return template

def exclude_digits(string):
    '''
    exclude the digits-domain words from partial constant
    '''
    pattern = r'\d'
    digits = re.findall(pattern, string)

    if len(digits)>=4:
        return True

    if len(digits)==0 or any(c.isupper() for c in string):
        return False

    return len(digits)/len(string) > 0.3

def correct_single_template_full(template, user_strings=None):
    """Apply all rules to process a template.

    DS (Double Space)
    BL (Boolean)
    US (User String)
    DG (Digit)
    PS (Path-like String)
    WV (Word concatenated with Variable)
    DV (Dot-separated Variables)
    CV (Consecutive Variables)

    """

    boolean = {'true', 'false'}
    default_strings = {'null', 'root', 'admin'}
    path_delimiters = {  # reduced set of delimiters for tokenizing for checking the path-like strings
        r'\s', r'\,', r'\!', r'\;', r'\:',
        r'\=', r'\|', r'\"', r'\'', r'\+',
        r'\[', r'\]', r'\(', r'\)', r'\{', r'\}'
    }
    token_delimiters = path_delimiters.union({  # all delimiters for tokenizing the remaining rules
        r'\.', r'\-', r'\@', r'\#', r'\$', r'\%', r'\&', 
    })

    if user_strings:
        default_strings = default_strings.union(user_strings)
    # default_strings = {}

    # apply DS
    # Note: this is not necessary while postprorcessing
    template = template.strip()
    template = re.sub(r'\s+', ' ', template)

    # apply PS
    p_tokens = re.split('(' + '|'.join(path_delimiters) + ')', template)
    new_p_tokens = []
    for p_token in p_tokens:
        # print(p_token)
        # if re.match(r'^(\/[^\/]+)+$', p_token) or re.match(r'^([a-zA-Z0-9-]+\.){2,}[a-zA-Z]+$', p_token):
        if re.match(r'^(\/[^\/]+)+\/?$', p_token) or re.match(r'^([a-z0-9-]+\.){2,}[a-z]+$', p_token):
            p_token = '<*>'
        if all(x in p_token for x in {'<*>', '.', '/'}):
            p_token = '<*>'
        # if re.search(r'(?=.*<\*>)(?=.*\.)(?=.*\/)', p_token):
        #     p_token = '<*>'
        
        new_p_tokens.append(p_token)
    template = ''.join(new_p_tokens)

    # tokenize for the remaining rules
    tokens = re.split('(' + '|'.join(token_delimiters) + ')', template)  # tokenizing while keeping delimiters
    new_tokens = []
    for token in tokens:
        # apply BL, US
        for to_replace in boolean.union(default_strings):
            # if token.lower() == to_replace.lower():
            if token == to_replace:
                token = '<*>'

        # apply DG
        # Note: hexadecimal num also appears a lot in the logs
        # if re.match(r'^\d+$', token) or re.match(r'\b0[xX][0-9a-fA-F]+\b', token):
        #     token = '<*>'
        if exclude_digits(token):
            token = '<*>'

        # apply WV
        if re.match(r'^[^\s\/]*<\*>[^\s\/]*$', token):
            # if token != '<*>/<*>':  # need to check this because `/` is not a deliminator
            token = '<*>'

        # collect the result
        new_tokens.append(token)

    # make the template using new_tokens
    template = ''.join(new_tokens)

    # Substitute consecutive variables only if separated with any delimiter including "." (DV)
    while True:
        prev = template
        template = re.sub(r'<\*>\.<\*>', '<*>', template)
        if prev == template:
            break

    # Substitute consecutive variables only if not separated with any delimiter including space (CV)
    # NOTE: this should be done at the end
    while True:
        prev = template
        template = re.sub(r'<\*><\*>', '<*>', template)
        if prev == template:
            break

    while "#<*>#" in template:
        template = template.replace("#<*>#", "<*>")

    while "<*>:<*>" in template:
        template = template.replace("<*>:<*>", "<*>")

    while "<*>/<*>" in template:
        template = template.replace("<*>/<*>", "<*>")

    while " #<*> " in template:
        template = template.replace(" #<*> ", " <*> ")

    while "<*>:<*>" in template:
        template = template.replace("<*>:<*>", "<*>")

    while "<*>#<*>" in template:
        template = template.replace("<*>#<*>", "<*>")

    while "<*>/<*>" in template:
        template = template.replace("<*>/<*>", "<*>")

    while "<*>@<*>" in template:
        template = template.replace("<*>@<*>", "<*>")

    while "<*>.<*>" in template:
        template = template.replace("<*>.<*>", "<*>")

    while ' "<*>" ' in template:
        template = template.replace(' "<*>" ', ' <*> ')

    while " '<*>' " in template:
        template = template.replace(" '<*>' ", " <*> ")

    while "<*><*>" in template:
        template = template.replace("<*><*>", "<*>")

    template = re.sub(r'<\*> [KGTM]?B\b', '<*>', template)

    return template


# if __name__ == '__main__':
    # import re
    # print(re.match(r'^(\/[^\/]+)+\/?$', "/"))
    # pattern = r'^([a-zA-Z0-9-]+\.){2,}[a-zA-Z]+$'
    # test_strings = [
    #     "example.com",
    #     "subdomain.example.com",
    #     "sub.subdomain.example.com",
    #     "sub-domain.example.co.uk",
    #     "example..com",
    #     "-example.com",
    #     "example-.com",
    #     "proxy.cse.cuhk.edu.hk",
    #     "example"
    # ]

    # for string in test_strings:
    #     if re.match(pattern, string):
    #         print(f"Matched: {string}")
    #     else:
    #         print(f"Did not match: {string}")

    # print(correct_single_template_full(
    #     "[ib_sm_bringup.c:577]: Force neighbor port (node=5ad000004b488, port=1, state=4) to DOWN because (1) 1st sweep or (2) role change."))
    # import pandas as pd
    # datasets = ['BGL', 'HDFS', 'HealthApp', 'OpenStack', 'OpenSSH', 'HPC', 'Zookeeper', 'Mac',
    #             'Hadoop', 'Android', 'Windows', 'Apache', 'Thunderbird', 'Spark', 'Linux', 'Proxifier']
    # num_2 = 0
    # for dataset in datasets:
    #     logs = []
    #     print("Processing", dataset)
    #     templates = pd.read_csv(
    #         f'../dataset/{dataset}/{dataset}_2k.log_templates_corrected.csv')
    #     num = 0
    #     for template,occur in zip(templates['EventTemplate'], templates['Occurrence']):
            # if template != correct_single_template(template):
            #     print("=" * 10)
            #     num+=occur
            #     print(dataset)
            #     print(template)
            #     print(correct_single_template(template))
            #     print()
            # correct_single_template(template)
            # if "<*> <*>" in template:
            #     print("=" * 10)
            #     print(dataset)
            #     print(template)
            #     print(correct_single_template(template))
            #     print()

# print(correct_single_template_full('Failed password for root from <*> port <*> ssh2'))
