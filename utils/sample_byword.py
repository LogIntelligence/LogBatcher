import random
import re

def extract_variables(log, template):
    # <*> -> (.*?)
    log = re.sub(r'\s+', ' ', log.strip())
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

    # length matters
    if len(log.split()) != len(cached_pair[0].split()):
        return None
    
    # DS
    log = re.sub(r'\s+', ' ', log.strip())

    pattern_parts = cached_pair[1].split("<*>")
    pattern_parts_escaped = [re.escape(part) for part in pattern_parts]
    regex_pattern = "(.*?)".join(pattern_parts_escaped)
    regex = "^" + regex_pattern + "$"  
    matches = re.search(regex, log)

    if not matches:
        return None  # 如果没有匹配，返回None

    # 生成新的template
    new_template_parts = []
    for index, part in enumerate(pattern_parts):
        new_template_parts.append(part)
        if index < len(matches.groups()):
            # 如果对应的匹配是空字符串，也在新模板中放置空字符串
            if matches.groups()[index] == '':
                new_template_parts.append('')
            else:
                new_template_parts.append('<*>')

    # 由于最后一个部分后面不需要加'<*>', 我们需要拼接处理后的模板部分
    new_template = ''.join(new_template_parts)
    return new_template



