from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import os
import random
import pandas as pd
import re
from sklearn.cluster import KMeans, DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from openai import OpenAI
import httpx
from tenacity import retry, stop_after_attempt, wait_random_exponential
from tqdm import tqdm
from evaluator import evaluate
from post_process import correct_single_template


class clusters:
    def __init__(self, label, logs, indexs, ground_truth):
        self.label = label
        self.logs = logs
        self.indexs = indexs
        self.ground_truth = ground_truth
    
    def remove_duplicate(self):
        self.logs = list(set(self.logs))

def tokenize(log_content, tokenize_pattern=r'[ ,]'):
    words = re.split(tokenize_pattern, log_content)
    list = ['/', 'kb', 'sec', 'byte', 'mb']
    for index, word in enumerate(words):
        if '=' in word:
            words[index] = word.split('=')[0]
        if re.search(r'\d', word):
            words[index] = ''
        if any(i in word.lower() for i in list):
            words[index] = ''
    words = [word for word in words if word]   # remove null
    return words


def vectorize(tokenized_logs):
    vectorizer = TfidfVectorizer(tokenizer=lambda x: x, lowercase=False)
    return vectorizer.fit_transform(tokenized_logs)


def cluster(vectorized_logs, num_clusters='10', cluster_method='kmeans'):
    if cluster_method == 'kmeans':
        cluster = KMeans(n_clusters=num_clusters)
    if cluster_method == 'dbscan':
        cluster = DBSCAN(eps=0.1, min_samples=5)
    cluster.fit(vectorized_logs)
    labels = cluster.labels_
    cluster_nums = max(labels) + 1
    return labels, cluster_nums
    

def reassign_clusters(labels, cluster_nums, tokenized_logs):
    mergerd_logs = []
    for tokenized_log in tokenized_logs:
        mergerd_logs.append(' '.join(tokenized_log))

    for i in range(len(labels)):
        if labels[i] == -1:
            for j in range(i+1, len(labels)):
                if labels[j] == -1 and mergerd_logs[i] == mergerd_logs[j]:
                    labels[j] = cluster_nums
            labels[i] = cluster_nums
            cluster_nums += 1
    return labels, cluster_nums

class Parser:
    def __init__(self, api_key, model='gpt-3.5-turbo-0125', using_proxy=True, cluster_method='dbscan', batch_num=50):
        self.api_key = api_key
        self.model = model
        self.cluster_method = cluster_method
        self.batch_num = batch_num
        self.random = True
        self.instruction_batch = '''You will be provided with some log messages. You should check if the giving log messages share the same template. If so, abstract variables with `{{placeholders}}` to extract the corresponding template.
        Print the input log's template delimited by backticks.'''  # The variables might be numbers, strings(name, url, path, data and time), or other types of data.
        self.instruciton_one_log = '''You will be provided with a log message delimited by backticks. You must abstract variables with `{{placeholders}}` to extract the corresponding template.
        Print the input log's template delimited by backticks.'''
        if using_proxy:
            self.client = OpenAI(
                # 3.5 https://4.0.996444.icu/v1
                base_url="https://oneapi.xty.app/v1",  # 中转url
                api_key=api_key,                      # api_key
                http_client=httpx.Client(
                    proxies="http://127.0.0.1:7890"  # 代理地址
                ),
            )
        else:
            self.client = OpenAI(
                base_url="https://oneapi.xty.app/v1", api_key=api_key)

    # @backoff.on_exception(backoff.expo, (openai.APIStatusError, openai.InternalServerError), max_tries=5)
    @retry(wait=wait_random_exponential(min=1, max=30), stop=stop_after_attempt(5))
    def chat(self, messages):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.0,
        )
        return response.choices[0].message.content.strip('\n')
    
    def get_responce(self, f, input):
        label, logs, indexs, ground_truth = input
        length = len(indexs)
        templates = []

        # remove duplicate
        # logs = list(set(logs))

        # remove duplicate conditionally
        # if len(list(set(logs))) == 1:
        #     logs = list(set(logs))
        
        if self.random:
            # seed = time.time()
            seed = 0
            random.seed(seed)
            random.shuffle(logs)

        for i in range(0, len(logs), self.batch_num):
            batch_logs = logs[i:i+self.batch_num]
            # if all logs's length is 1, and not contain any digit, return the log itself
            if all(len(re.split(' ', log)) == 1 and not any(char.isdigit() for char in log) for log in batch_logs):
                return batch_logs[0]

            messages = []
            if len(batch_logs) == 1:
                messages.append({"role": "system", "content": self.instruciton_one_log})
            else:
                messages.append({"role": "system", "content": self.instruction_batch})

            # demonstrations
                
            # # log + template
            # messages.append(
            #     {"role": "user", "content": '2017-07-02 15:46:41.445 ksfetch[32435/0x7fff79824000] [lvl=2] main() ksfetch fetching URL (<NSMutableURLRequest: 0x1005110b0> { URL: https://tools.google.com/service/update2?cup2hreq=53f725cf03f511fab16f19e789ce64aa1eed72395fc246e9f1100748325002f4&cup2key=7:1132320327 }) to folder:/tmp/KSOutOfProcessFetcher.YH2CjY1tnx/download'})
            # messages.append(
            #     {"role": "assistant", "content": '''`{{TIME}} ksfetch[{{ID_AND_ADDRESS}}] [lvl={{LEVEL}}] main() ksfetch fetching URL (<NSMutableURLRequest: {{ADDRESS}}> { URL: {{URL}} }) to folder:{{PATH}}`'''})
            # messages.append(
            #     {"role": "user", "content": '''Loading offline registry hive: SOFTWARE, into registry key '{bf1a281b-ad7b-4476-ac95-f47682990ce7}GLOBALROOT/Device/HarddiskVolumeShadowCopy2/Windows/System32/config/SOFTWARE' from path '\\?\GLOBALROOT\Device\HarddiskVolumeShadowCopy2\Windows\System32\config\SOFTWARE'.'''})
            # messages.append(
            #     {"role": "assistant", "content": '''`Loading offline registry hive: {{HIVE}}, into registry key '{{KEY}}' from path '{{PATH}}'.`'''})
            # messages.append(
            #     {"role": "user", "content": '''[CardDAVPlugin-ERROR] -getPrincipalInfo:[_controller supportsRequestCompressionAtURL:https://13957525385%40163.com@p28-contacts.icloud.com/874161398/principal/] Error Domain=NSURLErrorDomain Code=-1001 "The request timed out." UserInfo={NSUnderlyingError=0x7f9af3646900 {Error Domain=kCFErrorDomainCFNetwork Code=-1001 "The request timed out." UserInfo={NSErrorFailingURLStringKey=https://13957525385%40163.com@p28-contacts.icloud.com/874161398/principal/, NSErrorFailingURLKey=https://13957525385%40163.com@p28-contacts.icloud.com/874161398/principal/, _kCFStreamErrorCodeKey=-2102, _kCFStreamErrorDomainKey=4, NSLocalizedDescription=The request timed out.}}, NSErrorFailingURLStringKey=https://13957525385%40163.com@p28-contacts.icloud.com/874161398/principal/, NSErrorFailingURLKey=https://13957525385%40163.com@p28-contacts.icloud.com/874161398/principal/, _kCFStreamErrorDomainKey=4, _kCFStreamErrorCodeKey=-2102, NSLocalizedDescription=The request timed out.}'''})
            # messages.append(
            #     {"role": "assistant", "content": '''`[CardDAVPlugin-ERROR] -getPrincipalInfo:[_controller supportsRequestCompressionAtURL:{{URL}}] Error Domain=NSURLErrorDomain Code={{CODE}} "The request timed out." UserInfo={NSUnderlyingError={{ADDRESS}} {Error Domain=kCFErrorDomainCFNetwork Code={{CODE}} "The request timed out." UserInfo={NSErrorFailingURLStringKey={{URL}}, NSErrorFailingURLKey={{URL}}, _kCFStreamErrorCodeKey={{KEY}}, _kCFStreamErrorDomainKey={{KEY}}, NSLocalizedDescription=The request timed out.}}, NSErrorFailingURLStringKey={{URL}}, NSErrorFailingURLKey={{KEY}}, _kCFStreamErrorDomainKey={{KEY}}, _kCFStreamErrorCodeKey={{KEY}}, NSLocalizedDescription=The request timed out.}`'''})

            # # variable
            # messages.append(
            #     {"role": "user", "content": '''Kernel detected 35591540 integer alignment exceptions (35591533) iar 0x0023f108, dear 0x1feaa260 (35591534) iar 0x00265564, dear 0x1feaa1c0 (35591535) iar 0x00265574, dear 0x1feaa1e0 (35591536) iar 0x00265578, dear 0x1feaa200 (35591537) iar 0x00265588, dear 0x1feaa220 (35591538) iar 0x0026558c, dear 0x1feaa240 (35591539) iar 0x00265594, dear 0x1feaa260 (35591540) iar 0x00265598, dear 0x1feaa280'''})
            # messages.append(
            #     {"role": "assistant", "content": '''`Kernel detected {{COUNT}} integer alignment exceptions ({{ID}}) iar {{ADDRESS}}, dear {{ADDRESS}} ({{COUNT}}) iar {{ADDRESS}}, dear {{ADDRESS}} ({{COUNT}}) iar {{ADDRESS}}, dear {{ADDRESS}} ({{COUNT}}) iar {{ADDRESS}}, dear {{ADDRESS}} ({{COUNT}}) iar {{ADDRESS}}, dear {{ADDRESS}} ({{COUNT}}) iar {{ADDRESS}}, dear {{ADDRESS}} ({{COUNT}}) iar {{ADDRESS}}, dear {{ADDRESS}} ({{COUNT}}) iar {{ADDRESS}}, dear {{ADDRESS}}`'''})
            # messages.append(
            #     {"role": "user", "content": '''2017-07-07 10:54:41.875 GoogleSoftwareUpdateAgent[37924/0x7000002a0000] [lvl=2] -[KSUpdateCheckAction performAction] KSUpdateCheckAction starting update check for ticket(s): {( <KSTicket:0x100365950 productID=com.google.Chrome version=59.0.3071.115 xc=<KSPathExistenceChecker:0x10036e950 path=/Applications/Google Chrome.app> serverType=Omaha url=https://tools.google.com/service/update2 creationDate=2017-02-18 15:41:18 tagPath=/Applications/Google Chrome.app/Contents/Info.plist tagKey=KSChannelID brandPath=/Users/xpc/Library/Google/Google Chrome Brand.plist brandKey=KSBrandID versionPath=/Applications/Google Chrome.app/Contents/Info.plist versionKey=KSVersion cohort=1:1y5: cohortName=Stable ticketVersion=1 > )} Using server: <KSOmahaServer:0x100243f20 engine=<KSUpdateEngine:0x1007161b0> >'''})
            # messages.append(
            #     {"role": "assistant", "content": '''`{{TIME}} GoogleSoftwareUpdateAgent[{{ID}}] [lvl={{LEVEL}}] -[KSUpdateCheckAction performAction] KSUpdateCheckAction starting update check for ticket(s): {( <KSTicket:{{ADDRESS}} productID={{ID}} version={{VERSION}} xc=<KSPathExistenceChecker:{{ADDRESS}} path={{PATH}} serverType=Omaha url={{URL}} creationDate={{DATA}} tagPath={{PATH}} tagKey=KSChannelID brandPath={{PATH}} brandKey=KSBrandID versionPath={{PATH}} versionKey=KSVersion cohort={{ID}}: cohortName=Stable ticketVersion={{VERSION}} > )} Using server: <KSOmahaServer:{{ADDRESS}} engine=<KSUpdateEngine:{{ADDRESS}} >`'''})

            # manually
            messages.append({"role": "user", "content": ''''2017-07-02 15:46:41.445 application[0x7fff79824000] [lvl=2] main() Fetching URL: https://tools.google.com/service/update2?cup2hreq=53f725cf03f511fab16f19e789ce64aa1eed72395fc246e9f1100748325002f4&cup2key=7:1132320327 to folder: /tmp/ApplicationCache.xY4ZkL2vbn/download, ClientAddress: 198.51.100.14:62345, ServerEndpoint: api.server.com:443, RequestPath: /v1/data/process, VariableName: data_blk_042, Query user from exampleUser'''})
            messages.append({"role": "assistant", "content": '''`{{TIME}} application[{{ID}}] [lvl={{LEVEL}}] main() Fetching URL: {{URL}} to folder: {{PATH}}, ClientAddress: {{ADDRESS}}, ServerEndpoint: {{ADDRESS}}, RequestPath: {{PATH}}, VariableName: {{NAME}}, Query user from {{NAME}}`'''})

            # based the 10 types of variables
            messages.append({"role": "user", "content": '''Attempt 1445144423722 for object root10-local in domain ServerFileSystem has failed due to low computing resources (126MB LOWMEM available). Adding path spec: /mapreduce. Using configuration type 1. Observed change in network reachability, status now 2 (isReachable). Scheduled snapshot period at 10 seconds. Total of 23 ddr errors detected and corrected. Child workerEnv mod-jk in error state 7. Additional info: payload Data 0700 added to list of failed maps.'''})
            messages.append({"role": "assistant", "content": '''`Attempt {{ObejectID}} for object {{ObjectName}} in domain ServerFileSystem has failed due to low computing resources ({{ComputingResources}} LOWMEM available). Adding path spec: {{LocationIndicator}}. Using configuration type {{TypeIndicator}}. Observed change in network reachability, status now {{SwitchIndicator}} (isReachable). Scheduled snapshot period at {{TimeOrDuration}} seconds. Total of {{ObjectAmount}} ddr errors detected and corrected. Child workerEnv mod-jk in error state {{StatusCode}}. Additional info: payload Data {{OtherParameter}} added to list of failed maps.`'''})

            # batch logs to str
            prompt = ""
            length_prompt = 0
            for log in batch_logs:
                prompt += log + '\n'
                length_prompt += len(log)
            if length_prompt > 4096:
                prompt = ""
                for log in batch_logs[:5]:
                    prompt += log + '\n'
            messages.append({"role": "user", "content": prompt.strip('\n')})
            answer = self.chat(messages)
            template =  postprocessing(answer , isafter=False)
            if template != '':
                templates.append(template)

        final_tempalte, freq = choose(templates)

        f.write(f"---------------------------\n")
        f.write(f"cluster {label}: len={length}\n")
        f.write(f"{ground_truth} (ground truth)\n")
        # 打印结果
        for key, value in freq.items():
            f.write(f"{key}: {value}\n")
            # print(f"{key}: {value}")
        f.write(f"---------------------------\n")
        return final_tempalte




def postprocessing(response, isafter = False):

    response = response.strip().strip('\n')
    if "\n\n" in response:
        response = response.split("\n\n")[0]
    reg = re.compile("`([^`]+)`")
    tmps = reg.findall(response)
    tmps = [x.strip('\n').strip() for x in tmps]
    tmp = ''
    if len(tmps) == 1:
        tmp = tmps[0]
    if len(tmps) > 1:
        tmp = max(tmps, key=len)
    
    tmp = tmp.strip('\n').strip()
    tmp = re.sub(r'\{\{.*?\}\}', '<*>', tmp)
    template = tmp
    if not isafter:
        template = correct_single_template(template)
    if isafter:
        boolean = {'true', 'false'}
        default_strings = {'null', 'root', 'admin'}
        tokens = template.split(' ')
        for i in range(len(tokens)):
            if re.match(r'^\d+$', tokens[i]):
                tokens[i] = '<*>'
            for word in default_strings.union(boolean):
                tokens[i] = re.sub(r'(?i)(?<![a-z])' + word + r'(?![a-z])','<*>', tokens[i], flags=re.IGNORECASE)
            
            if tokens[i].count('<*>') >= 2:
                if tokens[i].startswith('/'):
                    tokens[i] = tokens[i][1:]
                # 保留前后的符号
                else:
                    prefix = '' if not re.match(
                        r'^[\[\]\.\:\,\/\']', tokens[i]) else tokens[i][0]
                    suffix = '' if not re.match(
                        r'.*[\[\]\.\:\,\/\']$', tokens[i]) else tokens[i][-1]
                    tokens[i] = prefix + '<*>' + suffix
        template = ' '.join(tokens)
    return template

    


def choose(list):

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
            freq = Counter(list)
            candidates = freq.most_common(len(freq))
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


def single_dataset_paring(dataset, output_dir, k = 10, cluster_method='kmeans', isConcurrent = True):
    print(f'Parsing {dataset}...')
    parser = Parser(
        api_key='sk-zY5LaAEd3EUdBVmKA75aDe77C9684c209b128b981826C043')
    df = pd.read_csv(f'dataset/{dataset}/{dataset}_2k.log_structured_corrected.csv')
    logs = df['Content'].tolist()

    # tokenize
    tokenized_logs = [tokenize(log) for log in logs]
    
    # cluster 1st
    labels, cluster_nums = cluster(vectorize(tokenized_logs), k, cluster_method)
    
    # reassign_clusters
    labels, cluster_nums = reassign_clusters(labels, cluster_nums, tokenized_logs)

    # output file
    os.makedirs(output_dir, exist_ok=True)
    f = open(output_dir + f'{dataset}.txt', 'w')

    outputs = [None for _ in range(2000)]
    
    inputs = []
    for i in range(cluster_nums):
        inputs.append([-1, [], [], '']) # label, logs, indexs, ground_truth
    for i, label in enumerate(labels):
        inputs[label][0] = label
        inputs[label][1].append(logs[i])
        inputs[label][2].append(i)
        if inputs[label][3] == '':
            inputs[label][3] = df['EventTemplate'][i]
    
    # Concurrent or not
    if isConcurrent:
        templates = []
        with ThreadPoolExecutor(max_workers=16) as executor:
            templates = list(
                tqdm(executor.map(parser.get_responce,[f]*len(inputs), inputs),
                    total=len(inputs)))
        for label, template in enumerate(templates):
            for index in inputs[label][2]:
                outputs[index] = template
    else:
        for label in range(cluster_nums):
            template = parser.get_responce(f, inputs[label])
            for index in inputs[label][2]:
                outputs[index] = template

    # write to file
    f.close()
    df['Output'] = outputs
    df[['Content', 'EventTemplate', 'Output']].to_csv(output_dir+ f'{dataset}.csv', index=False)
    evaluate(output_dir + f'{dataset}.csv', dataset)


# main
if __name__ == "__main__":
    datasets = ['BGL', 'HDFS', 'Linux', 'HealthApp', 'OpenStack', 'OpenSSH', 'Proxifier', 'HPC', 'Zookeeper', 'Mac',
                'Hadoop', 'Android', 'Windows', 'Apache', 'Thunderbird', 'Spark']
    # datasets = ['BGL', 'HDFS', 'Linux', 'HealthApp', 'OpenStack', 'OpenSSH', 'Proxifier', 'HPC', 'Zookeeper', 'Mac',
    #             'Hadoop',  'Apache', 'Thunderbird', 'Spark']  # 'Android', 'Windows' logpub
    # datasets = ['Windows', 'Apache', 'Thunderbird', 'Spark']
    # datasets = ['Thunderbird', 'Spark']
    cluster_nums = [132, 14, 143, 71, 56, 180, 14, 51, 54, 350, 115, 189, 57, 6, 194, 38]
    cluster_nums = [120, 14, 116, 75, 43,  26,  8, 46, 50, 341, 114, 158, 50, 6, 149, 36]
                 # [120, 14, 116, 75, 43,  26,  8, 46, 50, 341, 114, 158, 50, 6, 149, 36]
    output_dir = 'outputs/parser/Test/'
    for index, dataset in enumerate(datasets):
        k = cluster_nums[index]
        single_dataset_paring(dataset, output_dir, cluster_method='dbscan')
    
    # single_dataset_paring('Linux', cluster_method='dbscan')
